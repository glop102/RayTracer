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

// ------------------------------------------------------------------ private helpers

void ScreenshotMode::destroy_gbuf_images() {
    if (ss_albedo_view)  { vkDestroyImageView(device_, ss_albedo_view, nullptr);            ss_albedo_view  = VK_NULL_HANDLE; }
    if (ss_albedo_image) { vmaDestroyImage(allocator_, ss_albedo_image, ss_albedo_alloc);   ss_albedo_image = VK_NULL_HANDLE; }
    if (ss_normal_view)  { vkDestroyImageView(device_, ss_normal_view, nullptr);            ss_normal_view  = VK_NULL_HANDLE; }
    if (ss_normal_image) { vmaDestroyImage(allocator_, ss_normal_image, ss_normal_alloc);   ss_normal_image = VK_NULL_HANDLE; }
}

void ScreenshotMode::alloc_capture_buffers(VkContext& ctx) {
    if (accum_buf.buf)    vmaDestroyBuffer(ctx.allocator, accum_buf.buf,    accum_buf.alloc);
    if (readback_buf.buf) vmaDestroyBuffer(ctx.allocator, readback_buf.buf, readback_buf.alloc);
    destroy_gbuf_images();

    VkDeviceSize accum_size    = (VkDeviceSize)ss_width * ss_height * sizeof(double) * 4;
    VkDeviceSize readback_size = (VkDeviceSize)ss_width * ss_height * 4 * sizeof(float);

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
            throw std::runtime_error("HQ accum buffer allocation failed");
    }
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
            throw std::runtime_error("HQ readback buffer allocation failed");
    }

    // G-buffer images (albedo + normal) for OIDN denoising, always at ss_width×ss_height
    auto make_gbuf = [&](VkImage& img, VmaAllocation& alloc, VkImageView& view) {
        VkImageCreateInfo img_ci{};
        img_ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        img_ci.imageType     = VK_IMAGE_TYPE_2D;
        img_ci.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
        img_ci.extent        = {ss_width, ss_height, 1};
        img_ci.mipLevels     = 1;
        img_ci.arrayLayers   = 1;
        img_ci.samples       = VK_SAMPLE_COUNT_1_BIT;
        img_ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
        img_ci.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        if (vmaCreateImage(allocator_, &img_ci, &ai, &img, &alloc, nullptr) != VK_SUCCESS)
            throw std::runtime_error("Screenshot G-buffer image creation failed");

        VkImageViewCreateInfo view_ci{};
        view_ci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_ci.image            = img;
        view_ci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        view_ci.format           = VK_FORMAT_R32G32B32A32_SFLOAT;
        view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(device_, &view_ci, nullptr, &view) != VK_SUCCESS)
            throw std::runtime_error("Screenshot G-buffer image view creation failed");
    };
    make_gbuf(ss_albedo_image, ss_albedo_alloc, ss_albedo_view);
    make_gbuf(ss_normal_image, ss_normal_alloc, ss_normal_view);

    // Transition G-buffer images to GENERAL for storage writes
    VkCommandBuffer gcmd = ctx.begin_one_shot();
    img_barrier(gcmd, ss_albedo_image, 0, VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
    img_barrier(gcmd, ss_normal_image, 0, VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
    ctx.end_one_shot(gcmd);

    // Rebind accum buffer in the raygen set-1 descriptor (binding 0)
    VkDescriptorBufferInfo buf_info{accum_buf.buf, 0, accum_size};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet          = accum_set;
    w.dstBinding      = 0;
    w.descriptorCount = 1;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo     = &buf_info;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);

    // Bind G-buffer images to accum set bindings 1 and 2
    VkDescriptorImageInfo albedo_info{VK_NULL_HANDLE, ss_albedo_view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo normal_info{VK_NULL_HANDLE, ss_normal_view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet gbuf_writes[2]{};
    gbuf_writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    gbuf_writes[0].dstSet          = accum_set;
    gbuf_writes[0].dstBinding      = 1;
    gbuf_writes[0].descriptorCount = 1;
    gbuf_writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    gbuf_writes[0].pImageInfo      = &albedo_info;
    gbuf_writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    gbuf_writes[1].dstSet          = accum_set;
    gbuf_writes[1].dstBinding      = 2;
    gbuf_writes[1].descriptorCount = 1;
    gbuf_writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    gbuf_writes[1].pImageInfo      = &normal_info;
    vkUpdateDescriptorSets(device_, 2, gbuf_writes, 0, nullptr);

    // Rebind accum buffer in the resolve descriptor (binding 0)
    VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr.dstSet          = resolve_set;
    wr.dstBinding      = 0;
    wr.descriptorCount = 1;
    wr.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr.pBufferInfo     = &buf_info;
    vkUpdateDescriptorSets(device_, 1, &wr, 0, nullptr);
}

void ScreenshotMode::update_resolve_target(VkImageView view) {
    VkDescriptorImageInfo img_info{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet          = resolve_set;
    w.dstBinding      = 1;
    w.descriptorCount = 1;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w.pImageInfo      = &img_info;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);
}

void ScreenshotMode::destroy_ss_image() {
    if (ss_view)  { vkDestroyImageView(device_, ss_view, nullptr);          ss_view  = VK_NULL_HANDLE; }
    if (ss_image) { vmaDestroyImage(allocator_, ss_image, ss_alloc);        ss_image = VK_NULL_HANDLE; }
}

// ------------------------------------------------------------------ setup

void ScreenshotMode::setup(VkContext& ctx, VkExtent2D extent,
                            VkDescriptorSetLayout rt_output_layout,
                            VkImage color_image, VkImageView color_view) {
    device_         = ctx.device.device;
    allocator_      = ctx.allocator;
    pfn_trace_      = ctx.pfn_vkCmdTraceRaysKHR;
    width           = extent.width;
    height          = extent.height;
    ss_width        = extent.width;
    ss_height       = extent.height;
    rt_color_image_ = color_image;
    rt_color_view_  = color_view;

    // ---- accum descriptor set layout (set 1 for HQ raygen) ----
    // binding 0 = f64 accum SSBO, binding 1 = albedo image, binding 2 = normal image
    {
        VkDescriptorSetLayoutBinding bindings[3]{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 3;
        ci.pBindings    = bindings;
        if (vkCreateDescriptorSetLayout(device_, &ci, nullptr, &accum_dsl) != VK_SUCCESS)
            throw std::runtime_error("HQ accum DSL creation failed");

        VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  2},
        };
        VkDescriptorPoolCreateInfo pool_ci{};
        pool_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_ci.maxSets       = 1;
        pool_ci.poolSizeCount = 2;
        pool_ci.pPoolSizes    = pool_sizes;
        if (vkCreateDescriptorPool(device_, &pool_ci, nullptr, &accum_pool) != VK_SUCCESS)
            throw std::runtime_error("HQ accum pool creation failed");

        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = accum_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &accum_dsl;
        if (vkAllocateDescriptorSets(device_, &ai, &accum_set) != VK_SUCCESS)
            throw std::runtime_error("HQ accum set alloc failed");
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

        VkShaderModule mods[6] = {
            make_module(device_, rgen_spv),
            make_module(device_, rmiss_spv),
            make_module(device_, shadow_miss_spv),
            make_module(device_, rchit_spv),
            make_module(device_, glass_chit_spv),
            make_module(device_, glass_ahit_spv),
        };
        VkShaderStageFlagBits stage_bits[6] = {
            VK_SHADER_STAGE_RAYGEN_BIT_KHR,
            VK_SHADER_STAGE_MISS_BIT_KHR,
            VK_SHADER_STAGE_MISS_BIT_KHR,
            VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
            VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
            VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
        };
        VkPipelineShaderStageCreateInfo stages[6]{};
        for (int i = 0; i < 6; i++)
            stages[i] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                         nullptr, 0, stage_bits[i], mods[i], "main", nullptr};

        VkRayTracingShaderGroupCreateInfoKHR groups[5]{};
        for (auto& g : groups) {
            g.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            g.generalShader      = VK_SHADER_UNUSED_KHR;
            g.closestHitShader   = VK_SHADER_UNUSED_KHR;
            g.anyHitShader       = VK_SHADER_UNUSED_KHR;
            g.intersectionShader = VK_SHADER_UNUSED_KHR;
        }
        groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;           groups[0].generalShader    = 0;
        groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;           groups[1].generalShader    = 1;
        groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;           groups[2].generalShader    = 2;
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
        pipe_ci.stageCount                   = 6;
        pipe_ci.pStages                      = stages;
        pipe_ci.groupCount                   = 5;
        pipe_ci.pGroups                      = groups;
        pipe_ci.maxPipelineRayRecursionDepth = 1;
        pipe_ci.layout                       = hq_layout;
        if (ctx.pfn_vkCreateRayTracingPipelinesKHR(device_, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                                   1, &pipe_ci, nullptr, &hq_pipeline) != VK_SUCCESS)
            throw std::runtime_error("HQ RT pipeline creation failed");

        for (auto m : mods) vkDestroyShaderModule(device_, m, nullptr);

        // SBT
        uint32_t handle_size  = ctx.rt_pipeline_props.shaderGroupHandleSize;
        uint32_t handle_align = ctx.rt_pipeline_props.shaderGroupHandleAlignment;
        uint32_t base_align   = ctx.rt_pipeline_props.shaderGroupBaseAlignment;
        auto align_up = [](uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); };
        uint32_t stride        = align_up(handle_size, handle_align);
        uint32_t raygen_offset = 0;
        uint32_t miss_offset   = align_up(stride, base_align);
        uint32_t hit_offset    = align_up(miss_offset + 2 * stride, base_align);
        uint32_t sbt_size      = hit_offset + 2 * stride;

        std::vector<uint8_t> handles(5 * handle_size);
        if (ctx.pfn_vkGetRayTracingShaderGroupHandlesKHR(device_, hq_pipeline, 0, 5,
                                                         handles.size(), handles.data()) != VK_SUCCESS)
            throw std::runtime_error("HQ vkGetRayTracingShaderGroupHandlesKHR failed");

        VkBufferCreateInfo sbt_ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        sbt_ci.size  = sbt_size;
        sbt_ci.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VmaAllocationCreateInfo sbt_ai{};
        sbt_ai.usage = VMA_MEMORY_USAGE_AUTO;
        sbt_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo sbt_info{};
        if (vmaCreateBufferWithAlignment(ctx.allocator, &sbt_ci, &sbt_ai, base_align,
                                         &hq_sbt_buf, &hq_sbt_alloc, &sbt_info) != VK_SUCCESS)
            throw std::runtime_error("HQ SBT buffer creation failed");

        auto* data = static_cast<uint8_t*>(sbt_info.pMappedData);
        struct { uint32_t base, count; } regions[] = {
            {raygen_offset, 1}, {miss_offset, 2}, {hit_offset, 2}
        };
        uint32_t g = 0;
        for (auto [base, count] : regions)
            for (uint32_t i = 0; i < count; i++, g++)
                std::memcpy(data + base + i * stride, handles.data() + g * handle_size, handle_size);
        vmaFlushAllocation(ctx.allocator, hq_sbt_alloc, 0, VK_WHOLE_SIZE);

        VkBufferDeviceAddressInfo addr_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        addr_info.buffer = hq_sbt_buf;
        VkDeviceAddress sbt_addr = vkGetBufferDeviceAddress(device_, &addr_info);

        hq_raygen_region = {sbt_addr + raygen_offset, stride, stride};
        hq_miss_region   = {sbt_addr + miss_offset,   stride, 2 * stride};
        hq_hit_region    = {sbt_addr + hit_offset,    stride, 2 * stride};
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
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}
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
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                              nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr};
        VkComputePipelineCreateInfo pipe_ci{};
        pipe_ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipe_ci.stage  = stage;
        pipe_ci.layout = resolve_layout;
        if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipe_ci, nullptr, &resolve_pl) != VK_SUCCESS)
            throw std::runtime_error("Resolve compute pipeline creation failed");
        vkDestroyShaderModule(device_, mod, nullptr);
    }

    // Allocate initial capture buffers and point resolve at rt_output.image
    alloc_capture_buffers(ctx);
    update_resolve_target(rt_color_view_);
}

// ------------------------------------------------------------------ update_swapchain_size

void ScreenshotMode::update_swapchain_size(VkContext& ctx, VkExtent2D new_extent,
                                            VkImage color_image, VkImageView color_view) {
    active          = false;
    width           = new_extent.width;
    height          = new_extent.height;
    rt_color_image_ = color_image;
    rt_color_view_  = color_view;

    // If we're in same-res mode the resolve target tracks rt_output.image
    if (!custom_res)
        update_resolve_target(rt_color_view_);
}

// ------------------------------------------------------------------ begin

void ScreenshotMode::begin(VkContext& ctx, uint32_t n_samples, uint32_t ss_w, uint32_t ss_h) {
    if (ss_w == 0) ss_w = width;
    if (ss_h == 0) ss_h = height;

    // Validate against hardware limits before allocating anything.
    // The accum SSBO (dvec4 per pixel = 32 bytes) must fit within maxStorageBufferRange.
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.physical_device.physical_device, &props);
    VkDeviceSize max_ssbo  = props.limits.maxStorageBufferRange;
    VkDeviceSize accum_req = (VkDeviceSize)ss_w * ss_h * sizeof(double) * 4;
    if (accum_req > max_ssbo) {
        fprintf(stderr,
            "Screenshot: requested resolution %u×%u requires %.1f GB for the accumulation buffer "
            "but maxStorageBufferRange is %.1f GB. Reduce the resolution.\n",
            ss_w, ss_h,
            accum_req / 1073741824.0,
            max_ssbo  / 1073741824.0);
        return;
    }

    bool dims_changed = (ss_w != ss_width || ss_h != ss_height);
    ss_width  = ss_w;
    ss_height = ss_h;
    custom_res = (ss_width != width || ss_height != height);

    if (dims_changed)
        alloc_capture_buffers(ctx);

    if (custom_res) {
        // Allocate (or reallocate) private output image at screenshot resolution
        destroy_ss_image();

        VkImageCreateInfo img_ci{};
        img_ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        img_ci.imageType     = VK_IMAGE_TYPE_2D;
        img_ci.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
        img_ci.extent        = {ss_width, ss_height, 1};
        img_ci.mipLevels     = 1;
        img_ci.arrayLayers   = 1;
        img_ci.samples       = VK_SAMPLE_COUNT_1_BIT;
        img_ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
        img_ci.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo img_ai{};
        img_ai.usage = VMA_MEMORY_USAGE_AUTO;
        img_ai.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        if (vmaCreateImage(allocator_, &img_ci, &img_ai, &ss_image, &ss_alloc, nullptr) != VK_SUCCESS)
            throw std::runtime_error("Screenshot output image creation failed");

        VkImageViewCreateInfo view_ci{};
        view_ci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_ci.image            = ss_image;
        view_ci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        view_ci.format           = VK_FORMAT_R32G32B32A32_SFLOAT;
        view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(device_, &view_ci, nullptr, &ss_view) != VK_SUCCESS)
            throw std::runtime_error("Screenshot output image view creation failed");

        // Transition to GENERAL for storage image writes
        VkCommandBuffer cmd = ctx.begin_one_shot();
        img_barrier(cmd, ss_image, 0, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        ctx.end_one_shot(cmd);

        update_resolve_target(ss_view);
    } else {
        destroy_ss_image();
        update_resolve_target(rt_color_view_);
    }

    samples_done = 0;
    target       = n_samples;
    active       = true;

    // Zero accumulation buffer
    VkCommandBuffer cmd = ctx.begin_one_shot();
    vkCmdFillBuffer(cmd, accum_buf.buf, 0, VK_WHOLE_SIZE, 0);
    buf_barrier(cmd, accum_buf.buf,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
    ctx.end_one_shot(cmd);
}

// ------------------------------------------------------------------ record_sample

void ScreenshotMode::record_sample(VkCommandBuffer cmd, VkDescriptorSet rt_output_set,
                                    const RtCameraPush& push) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, hq_pipeline);
    vkCmdPushConstants(cmd, hq_layout, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(push), &push);

    VkDescriptorSet sets[] = {rt_output_set, accum_set};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            hq_layout, 0, 2, sets, 0, nullptr);

    pfn_trace_(cmd, &hq_raygen_region, &hq_miss_region,
               &hq_hit_region, &hq_callable_region, ss_width, ss_height, 1);
}

// ------------------------------------------------------------------ resolve

void ScreenshotMode::resolve(VkContext& ctx) {
    VkCommandBuffer cmd = ctx.begin_one_shot();

    buf_barrier(cmd, accum_buf.buf,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resolve_pl);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resolve_layout,
                            0, 1, &resolve_set, 0, nullptr);

    struct { uint32_t sample_count, width, height; } push{target, ss_width, ss_height};
    vkCmdPushConstants(cmd, resolve_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (ss_width + 7) / 8, (ss_height + 7) / 8, 1);

    // Make resolve writes visible for subsequent readback or denoiser
    VkImage resolved_image = custom_res ? ss_image : rt_color_image_;
    img_barrier(cmd, resolved_image,
                VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT);

    ctx.end_one_shot(cmd);
}

// ------------------------------------------------------------------ save_png

void ScreenshotMode::save_png(VkContext& ctx, VkImage src_image, VkImageLayout src_layout,
                               const std::string& path) {
    VkCommandBuffer cmd = ctx.begin_one_shot();

    bool need_transition = (src_layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    if (need_transition) {
        img_barrier(cmd, src_image,
                    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    src_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {ss_width, ss_height, 1};
    vkCmdCopyImageToBuffer(cmd, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback_buf.buf, 1, &region);

    if (need_transition) {
        img_barrier(cmd, src_image,
                    VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
    }

    ctx.end_one_shot(cmd);

    vmaInvalidateAllocation(ctx.allocator, readback_buf.alloc, 0, VK_WHOLE_SIZE);
    VmaAllocationInfo alloc_info{};
    vmaGetAllocationInfo(ctx.allocator, readback_buf.alloc, &alloc_info);
    const float* rgba = static_cast<const float*>(alloc_info.pMappedData);

    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) return;

    png_structp png  = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop   info = png_create_info_struct(png);
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return;
    }
    png_init_io(png, fp);
    png_set_IHDR(png, info, ss_width, ss_height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    std::vector<uint8_t> row(ss_width * 3);
    for (uint32_t y = 0; y < ss_height; y++) {
        for (uint32_t x = 0; x < ss_width; x++) {
            const float* px = rgba + (y * ss_width + x) * 4;
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
    destroy_ss_image();
    destroy_gbuf_images();

    if (resolve_pl)     vkDestroyPipeline(device_, resolve_pl, nullptr);
    if (resolve_layout) vkDestroyPipelineLayout(device_, resolve_layout, nullptr);
    if (resolve_pool)   vkDestroyDescriptorPool(device_, resolve_pool, nullptr);
    if (resolve_dsl)    vkDestroyDescriptorSetLayout(device_, resolve_dsl, nullptr);

    if (hq_sbt_buf)  vmaDestroyBuffer(ctx.allocator, hq_sbt_buf, hq_sbt_alloc);
    if (hq_pipeline) vkDestroyPipeline(device_, hq_pipeline, nullptr);
    if (hq_layout)   vkDestroyPipelineLayout(device_, hq_layout, nullptr);

    if (accum_pool) vkDestroyDescriptorPool(device_, accum_pool, nullptr);
    if (accum_dsl)  vkDestroyDescriptorSetLayout(device_, accum_dsl, nullptr);

    if (accum_buf.buf)    vmaDestroyBuffer(ctx.allocator, accum_buf.buf,    accum_buf.alloc);
    if (readback_buf.buf) vmaDestroyBuffer(ctx.allocator, readback_buf.buf, readback_buf.alloc);

    *this = {};
}
