#include "screenshot.h"
#include "vk_context.h"
#include "shader_compiler.h"

#include <png.h>
#include <cmath>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <vector>

// ------------------------------------------------------------------ helpers

static void buf_barrier(VkCommandBuffer cmd, VkBuffer buf,
                         VkAccessFlags src, VkAccessFlags dst,
                         VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    VkBufferMemoryBarrier b{};
    b.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.buffer        = buf;
    b.offset        = 0;
    b.size          = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 1, &b, 0, nullptr);
}

static void img_barrier(VkCommandBuffer cmd, VkImage image,
                         VkAccessFlags src, VkAccessFlags dst,
                         VkImageLayout old_layout, VkImageLayout new_layout,
                         VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier b{};
    b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask       = src;
    b.dstAccessMask       = dst;
    b.oldLayout           = old_layout;
    b.newLayout           = new_layout;
    b.image               = image;
    b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

static uint8_t linear_to_srgb_u8(float x) {
    x = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
    float s = x <= 0.0031308f ? 12.92f * x
                              : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
    return static_cast<uint8_t>(std::lround(s * 255.0f));
}

// ------------------------------------------------------------------ setup

void ScreenshotMode::setup(VkContext& ctx, VkExtent2D extent,
                            VkDescriptorSetLayout rt_output_layout,
                            VkImage color_image, VkImageView color_view) {
    device_      = ctx.device.device;
    allocator_   = ctx.allocator;
    pfn_trace_   = ctx.pfn_vkCmdTraceRaysKHR;
    width        = extent.width;
    height       = extent.height;
    color_image_ = color_image;

    // ---- accum buffer (dvec4 per pixel = 32 bytes) ----
    VkDeviceSize accum_size = (VkDeviceSize)width * height * sizeof(double) * 4;
    {
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size  = accum_size;
        ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        accum_buf.size = accum_size;
        if (vmaCreateBuffer(ctx.allocator, &ci, &ai, &accum_buf.buf, &accum_buf.alloc, nullptr) != VK_SUCCESS)
            throw std::runtime_error("HQ accum buffer creation failed");
    }

    // ---- readback staging buffer ----
    VkDeviceSize readback_size = (VkDeviceSize)width * height * 4 * sizeof(float);
    {
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size  = readback_size;
        ci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                   VMA_ALLOCATION_CREATE_MAPPED_BIT;
        readback_buf.size = readback_size;
        if (vmaCreateBuffer(ctx.allocator, &ci, &ai, &readback_buf.buf, &readback_buf.alloc, nullptr) != VK_SUCCESS)
            throw std::runtime_error("HQ readback buffer creation failed");
    }

    // ---- accum descriptor set layout (set 1 for HQ raygen) ----
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings    = &b;
        if (vkCreateDescriptorSetLayout(device_, &ci, nullptr, &accum_dsl) != VK_SUCCESS)
            throw std::runtime_error("HQ accum DSL creation failed");
    }
    {
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.maxSets       = 1;
        ci.poolSizeCount = 1;
        ci.pPoolSizes    = &ps;
        if (vkCreateDescriptorPool(device_, &ci, nullptr, &accum_pool) != VK_SUCCESS)
            throw std::runtime_error("HQ accum pool creation failed");

        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = accum_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &accum_dsl;
        if (vkAllocateDescriptorSets(device_, &ai, &accum_set) != VK_SUCCESS)
            throw std::runtime_error("HQ accum set alloc failed");

        VkDescriptorBufferInfo buf_info{accum_buf.buf, 0, accum_size};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet          = accum_set;
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo     = &buf_info;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }

    // ---- HQ ray tracing pipeline ----
    {
        auto shader_dir = find_shader_dir();

        auto rgen_spv        = compile_glsl(read_file(shader_dir / "raygen_hq.rgen"),          "raygen_hq.rgen",         shaderc_raygen_shader,       shader_dir);
        auto rmiss_spv       = compile_glsl(read_file(shader_dir / "miss.rmiss"),              "miss.rmiss",             shaderc_miss_shader,         shader_dir);
        auto shadow_miss_spv = compile_glsl(read_file(shader_dir / "shadow_miss.rmiss"),       "shadow_miss.rmiss",      shaderc_miss_shader,         shader_dir);
        auto rchit_spv       = compile_glsl(read_file(shader_dir / "closest_hit.rchit"),       "closest_hit.rchit",      shaderc_closesthit_shader,   shader_dir);
        auto glass_chit_spv  = compile_glsl(read_file(shader_dir / "glass_closest_hit.rchit"),"glass_closest_hit.rchit",shaderc_closesthit_shader,   shader_dir);
        auto glass_ahit_spv  = compile_glsl(read_file(shader_dir / "glass_any_hit.rahit"),     "glass_any_hit.rahit",    shaderc_anyhit_shader,       shader_dir);

        VkShaderModule rgen_mod        = make_module(device_, rgen_spv);
        VkShaderModule rmiss_mod       = make_module(device_, rmiss_spv);
        VkShaderModule shadow_miss_mod = make_module(device_, shadow_miss_spv);
        VkShaderModule rchit_mod       = make_module(device_, rchit_spv);
        VkShaderModule glass_chit_mod  = make_module(device_, glass_chit_spv);
        VkShaderModule glass_ahit_mod  = make_module(device_, glass_ahit_spv);

        auto make_stage = [](VkShaderStageFlagBits stage, VkShaderModule mod) {
            return VkPipelineShaderStageCreateInfo{
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                nullptr, 0, stage, mod, "main", nullptr};
        };
        VkPipelineShaderStageCreateInfo stages[] = {
            make_stage(VK_SHADER_STAGE_RAYGEN_BIT_KHR,      rgen_mod),
            make_stage(VK_SHADER_STAGE_MISS_BIT_KHR,        rmiss_mod),
            make_stage(VK_SHADER_STAGE_MISS_BIT_KHR,        shadow_miss_mod),
            make_stage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, rchit_mod),
            make_stage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, glass_chit_mod),
            make_stage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR,     glass_ahit_mod),
        };

        VkRayTracingShaderGroupCreateInfoKHR groups[5]{};
        for (auto& g : groups) {
            g.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            g.closestHitShader   = VK_SHADER_UNUSED_KHR;
            g.anyHitShader       = VK_SHADER_UNUSED_KHR;
            g.intersectionShader = VK_SHADER_UNUSED_KHR;
            g.generalShader      = VK_SHADER_UNUSED_KHR;
        }
        groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR; groups[0].generalShader = 0;
        groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR; groups[1].generalShader = 1;
        groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR; groups[2].generalShader = 2;
        groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR; groups[3].closestHitShader = 3;
        groups[4].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        groups[4].closestHitShader = 4; groups[4].anyHitShader = 5;

        VkDescriptorSetLayout set_layouts[] = {rt_output_layout, accum_dsl};
        VkPushConstantRange push_range{VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(RtCameraPush)};

        VkPipelineLayoutCreateInfo layout_ci{};
        layout_ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_ci.setLayoutCount         = 2;
        layout_ci.pSetLayouts            = set_layouts;
        layout_ci.pushConstantRangeCount = 1;
        layout_ci.pPushConstantRanges    = &push_range;
        if (vkCreatePipelineLayout(device_, &layout_ci, nullptr, &hq_layout) != VK_SUCCESS)
            throw std::runtime_error("HQ pipeline layout creation failed");

        VkRayTracingPipelineCreateInfoKHR pipe_ci{};
        pipe_ci.sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        pipe_ci.stageCount                   = static_cast<uint32_t>(std::size(stages));
        pipe_ci.pStages                      = stages;
        pipe_ci.groupCount                   = static_cast<uint32_t>(std::size(groups));
        pipe_ci.pGroups                      = groups;
        pipe_ci.maxPipelineRayRecursionDepth = 1;
        pipe_ci.layout                       = hq_layout;
        if (ctx.pfn_vkCreateRayTracingPipelinesKHR(device_, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                                   1, &pipe_ci, nullptr, &hq_pipeline) != VK_SUCCESS)
            throw std::runtime_error("HQ RT pipeline creation failed");

        vkDestroyShaderModule(device_, rgen_mod,        nullptr);
        vkDestroyShaderModule(device_, rmiss_mod,       nullptr);
        vkDestroyShaderModule(device_, shadow_miss_mod, nullptr);
        vkDestroyShaderModule(device_, rchit_mod,       nullptr);
        vkDestroyShaderModule(device_, glass_chit_mod,  nullptr);
        vkDestroyShaderModule(device_, glass_ahit_mod,  nullptr);

        // SBT (same structure as regular pipeline)
        uint32_t handle_size  = ctx.rt_pipeline_props.shaderGroupHandleSize;
        uint32_t handle_align = ctx.rt_pipeline_props.shaderGroupHandleAlignment;
        uint32_t base_align   = ctx.rt_pipeline_props.shaderGroupBaseAlignment;
        auto align_up = [](uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); };
        uint32_t handle_stride = align_up(handle_size, handle_align);
        constexpr uint32_t MISS_COUNT = 2, HIT_COUNT = 2;
        uint32_t raygen_offset = 0;
        uint32_t miss_offset   = align_up(handle_stride, base_align);
        uint32_t hit_offset    = align_up(miss_offset + MISS_COUNT * handle_stride, base_align);
        uint32_t sbt_size      = hit_offset + HIT_COUNT * handle_stride;

        std::vector<uint8_t> handles(5 * handle_size);
        if (ctx.pfn_vkGetRayTracingShaderGroupHandlesKHR(device_, hq_pipeline, 0, 5,
                                                         handles.size(), handles.data()) != VK_SUCCESS)
            throw std::runtime_error("HQ vkGetRayTracingShaderGroupHandlesKHR failed");

        VkBufferCreateInfo sbt_ci{};
        sbt_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        sbt_ci.size  = sbt_size;
        sbt_ci.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VmaAllocationCreateInfo sbt_ai{};
        sbt_ai.usage = VMA_MEMORY_USAGE_AUTO;
        sbt_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo sbt_info{};
        if (vmaCreateBufferWithAlignment(ctx.allocator, &sbt_ci, &sbt_ai,
                                         base_align, &hq_sbt_buf, &hq_sbt_alloc, &sbt_info) != VK_SUCCESS)
            throw std::runtime_error("HQ SBT buffer creation failed");

        auto* data = static_cast<uint8_t*>(sbt_info.pMappedData);
        struct { uint32_t base; uint32_t count; } regions[] = {
            {raygen_offset, 1}, {miss_offset, MISS_COUNT}, {hit_offset, HIT_COUNT}
        };
        uint32_t g = 0;
        for (auto [base, count] : regions)
            for (uint32_t i = 0; i < count; i++, g++)
                std::memcpy(data + base + i * handle_stride, handles.data() + g * handle_size, handle_size);
        vmaFlushAllocation(ctx.allocator, hq_sbt_alloc, 0, VK_WHOLE_SIZE);

        VkBufferDeviceAddressInfo addr_info{};
        addr_info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addr_info.buffer = hq_sbt_buf;
        VkDeviceAddress sbt_addr = vkGetBufferDeviceAddress(device_, &addr_info);

        hq_raygen_region = {sbt_addr + raygen_offset, handle_stride, handle_stride};
        hq_miss_region   = {sbt_addr + miss_offset,   handle_stride, MISS_COUNT * handle_stride};
        hq_hit_region    = {sbt_addr + hit_offset,    handle_stride, HIT_COUNT  * handle_stride};
    }

    // ---- Resolve compute pipeline ----
    {
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo dsl_ci{};
        dsl_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsl_ci.bindingCount = 2;
        dsl_ci.pBindings    = bindings;
        if (vkCreateDescriptorSetLayout(device_, &dsl_ci, nullptr, &resolve_dsl) != VK_SUCCESS)
            throw std::runtime_error("Resolve DSL creation failed");

        VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1},
        };
        VkDescriptorPoolCreateInfo pool_ci{};
        pool_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_ci.maxSets       = 1;
        pool_ci.poolSizeCount = 2;
        pool_ci.pPoolSizes    = pool_sizes;
        if (vkCreateDescriptorPool(device_, &pool_ci, nullptr, &resolve_pool) != VK_SUCCESS)
            throw std::runtime_error("Resolve pool creation failed");

        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = resolve_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &resolve_dsl;
        if (vkAllocateDescriptorSets(device_, &ai, &resolve_set) != VK_SUCCESS)
            throw std::runtime_error("Resolve set alloc failed");

        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * sizeof(uint32_t)};
        VkPipelineLayoutCreateInfo layout_ci{};
        layout_ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_ci.setLayoutCount         = 1;
        layout_ci.pSetLayouts            = &resolve_dsl;
        layout_ci.pushConstantRangeCount = 1;
        layout_ci.pPushConstantRanges    = &push;
        if (vkCreatePipelineLayout(device_, &layout_ci, nullptr, &resolve_layout) != VK_SUCCESS)
            throw std::runtime_error("Resolve pipeline layout creation failed");

        auto shader_dir = find_shader_dir();
        auto spv = compile_glsl(read_file(shader_dir / "hq_resolve.comp"), "hq_resolve.comp",
                                shaderc_compute_shader, shader_dir);
        VkShaderModule mod = make_module(device_, spv);

        VkPipelineShaderStageCreateInfo stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr};
        VkComputePipelineCreateInfo pipe_ci{};
        pipe_ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipe_ci.stage  = stage;
        pipe_ci.layout = resolve_layout;
        if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipe_ci, nullptr, &resolve_pl) != VK_SUCCESS)
            throw std::runtime_error("Resolve compute pipeline creation failed");
        vkDestroyShaderModule(device_, mod, nullptr);
    }

    // Write resolve descriptor set (accum buffer + color image)
    {
        VkDescriptorBufferInfo buf_info{accum_buf.buf, 0, accum_buf.size};
        VkDescriptorImageInfo  img_info{VK_NULL_HANDLE, color_view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet writes[2]{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = resolve_set;
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo     = &buf_info;
        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = resolve_set;
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo      = &img_info;
        vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
    }
}

void ScreenshotMode::update_color_image(VkContext& ctx, VkExtent2D new_extent,
                                         VkImage color_image, VkImageView color_view) {
    color_image_ = color_image;
    active       = false;  // abort any in-progress capture

    bool dims_changed = (new_extent.width != width || new_extent.height != height);
    if (dims_changed) {
        width  = new_extent.width;
        height = new_extent.height;

        // Recreate accum buffer for new resolution
        if (accum_buf.buf) vmaDestroyBuffer(ctx.allocator, accum_buf.buf, accum_buf.alloc);
        VkDeviceSize accum_size = (VkDeviceSize)width * height * sizeof(double) * 4;
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size  = accum_size;
        ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        accum_buf.size = accum_size;
        if (vmaCreateBuffer(ctx.allocator, &ci, &ai, &accum_buf.buf, &accum_buf.alloc, nullptr) != VK_SUCCESS)
            throw std::runtime_error("HQ accum buffer resize failed");

        // Recreate readback buffer
        if (readback_buf.buf) vmaDestroyBuffer(ctx.allocator, readback_buf.buf, readback_buf.alloc);
        VkDeviceSize readback_size = (VkDeviceSize)width * height * 4 * sizeof(float);
        ci.size  = readback_size;
        ci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        readback_buf.size = readback_size;
        if (vmaCreateBuffer(ctx.allocator, &ci, &ai, &readback_buf.buf, &readback_buf.alloc, nullptr) != VK_SUCCESS)
            throw std::runtime_error("HQ readback buffer resize failed");

        // Rebind accum buffer in the accum descriptor set (set 1 for raygen)
        VkDescriptorBufferInfo buf_info_accum{accum_buf.buf, 0, accum_size};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet          = accum_set;
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo     = &buf_info_accum;
        vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
    }

    // Always rebind color image view in resolve descriptor set
    VkDescriptorBufferInfo buf_info{accum_buf.buf, 0, accum_buf.size};
    VkDescriptorImageInfo  img_info{VK_NULL_HANDLE, color_view, VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = resolve_set;
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo     = &buf_info;

    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = resolve_set;
    writes[1].dstBinding      = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo      = &img_info;

    vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
}

// ------------------------------------------------------------------ begin

void ScreenshotMode::begin(VkContext& ctx, uint32_t n_samples) {
    samples_done = 0;
    target       = n_samples;
    active       = true;

    // Zero the accumulation buffer so partial results don't bleed in
    VkCommandBuffer cmd = ctx.begin_one_shot();
    vkCmdFillBuffer(cmd, accum_buf.buf, 0, VK_WHOLE_SIZE, 0);
    buf_barrier(cmd, accum_buf.buf,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
    ctx.end_one_shot(cmd);
}

// ------------------------------------------------------------------ record_sample

void ScreenshotMode::record_sample(VkCommandBuffer cmd, VkDescriptorSet rt_output_set,
                                    const RtCameraPush& push, uint32_t w, uint32_t h) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, hq_pipeline);

    vkCmdPushConstants(cmd, hq_layout,
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(push), &push);

    VkDescriptorSet sets[] = {rt_output_set, accum_set};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            hq_layout, 0, 2, sets, 0, nullptr);

    pfn_trace_(cmd,
        &hq_raygen_region, &hq_miss_region,
        &hq_hit_region,    &hq_callable_region,
        w, h, 1);
}

// ------------------------------------------------------------------ resolve

void ScreenshotMode::resolve(VkContext& ctx) {
    VkCommandBuffer cmd = ctx.begin_one_shot();

    // Ensure raygen writes to accum_buf are visible to compute
    buf_barrier(cmd, accum_buf.buf,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resolve_pl);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resolve_layout,
                            0, 1, &resolve_set, 0, nullptr);

    struct { uint32_t sample_count, width, height; } push{target, width, height};
    vkCmdPushConstants(cmd, resolve_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t gx = (width  + 7) / 8;
    uint32_t gy = (height + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    // Transition color image to GENERAL so denoiser/readback can use it
    img_barrier(cmd, color_image_,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT);

    ctx.end_one_shot(cmd);
}

// ------------------------------------------------------------------ save_png

void ScreenshotMode::save_png(VkContext& ctx, VkImage src_image, VkImageLayout src_layout,
                               const std::string& path) {
    VkCommandBuffer cmd = ctx.begin_one_shot();

    // Transition to TRANSFER_SRC if needed
    bool need_transition = (src_layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    if (need_transition) {
        img_barrier(cmd, src_image,
                    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    src_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    VkBufferImageCopy region{};
    region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent       = {width, height, 1};
    vkCmdCopyImageToBuffer(cmd, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback_buf.buf, 1, &region);

    // Transition back to GENERAL
    if (need_transition) {
        img_barrier(cmd, src_image,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
    }

    ctx.end_one_shot(cmd);  // waits for GPU

    // Invalidate host cache before reading (required for non-coherent memory)
    vmaInvalidateAllocation(ctx.allocator, readback_buf.alloc, 0, VK_WHOLE_SIZE);

    VmaAllocationInfo alloc_info{};
    vmaGetAllocationInfo(ctx.allocator, readback_buf.alloc, &alloc_info);
    const float* rgba = static_cast<const float*>(alloc_info.pMappedData);

    // Write PNG
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) { vmaUnmapMemory(ctx.allocator, readback_buf.alloc); return; }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop   info = png_create_info_struct(png);
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return;
    }
    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    std::vector<uint8_t> row(width * 3);
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            const float* px = rgba + (y * width + x) * 4;
            row[x * 3 + 0] = linear_to_srgb_u8(px[0]);
            row[x * 3 + 1] = linear_to_srgb_u8(px[1]);
            row[x * 3 + 2] = linear_to_srgb_u8(px[2]);
        }
        png_write_row(png, row.data());
    }
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(fp);

}

// ------------------------------------------------------------------ destroy

void ScreenshotMode::destroy(VkContext& ctx) {
    if (resolve_pl)     vkDestroyPipeline(device_, resolve_pl, nullptr);
    if (resolve_layout) vkDestroyPipelineLayout(device_, resolve_layout, nullptr);
    if (resolve_pool)   vkDestroyDescriptorPool(device_, resolve_pool, nullptr);
    if (resolve_dsl)    vkDestroyDescriptorSetLayout(device_, resolve_dsl, nullptr);

    if (hq_sbt_buf)  vmaDestroyBuffer(ctx.allocator, hq_sbt_buf, hq_sbt_alloc);
    if (hq_pipeline) vkDestroyPipeline(device_, hq_pipeline, nullptr);
    if (hq_layout)   vkDestroyPipelineLayout(device_, hq_layout, nullptr);

    if (accum_pool) vkDestroyDescriptorPool(device_, accum_pool, nullptr);
    if (accum_dsl)  vkDestroyDescriptorSetLayout(device_, accum_dsl, nullptr);

    if (accum_buf.buf)    vmaDestroyBuffer(ctx.allocator, accum_buf.buf, accum_buf.alloc);
    if (readback_buf.buf) vmaDestroyBuffer(ctx.allocator, readback_buf.buf, readback_buf.alloc);

    *this = {};
}
