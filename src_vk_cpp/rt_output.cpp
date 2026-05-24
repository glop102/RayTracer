#include "rt_output.h"
#include "vk_context.h"

#include <stdexcept>
#include <vector>

static constexpr VkFormat STORAGE_FORMAT = VK_FORMAT_R32G32B32A32_SFLOAT;

RtOutput::RtOutput(VkContext& ctx, VkExtent2D extent, VkAccelerationStructureKHR tlas,
                   std::span<const GpuBuffer> ssbos,
                   std::span<const VkImageView> tex_views,
                   VkSampler sampler) {
    device     = ctx.device.device;
    allocator  = ctx.allocator;
    ssbos_     = {ssbos.begin(), ssbos.end()};
    tex_views_ = {tex_views.begin(), tex_views.end()};
    sampler_   = sampler;

    uint32_t num_ssbos = static_cast<uint32_t>(ssbos_.size());
    uint32_t num_tex   = static_cast<uint32_t>(tex_views_.size());

    // Binding numbers (must match shader hardcoded values).
    // SSBOs occupy [2 .. 2+num_ssbos-1].  G-buffer images follow with a 1-slot gap,
    // then the variable-count texture array at the highest binding.
    uint32_t bind_albedo = 2 + num_ssbos + 1;  // = 7 with 4 SSBOs
    uint32_t bind_normal = 2 + num_ssbos + 2;  // = 8
    uint32_t bind_tex    = 2 + num_ssbos + 3;  // = 9
    static constexpr uint32_t MAX_TEX = 65536;

    // ------------------------------------------------------------------ Descriptor set layout
    // Array entries: 1 TLAS + 1 colour image + N SSBOs + 2 G-buffer images + 1 texture array.
    uint32_t total_bindings = 2 + num_ssbos + 3;
    std::vector<VkDescriptorSetLayoutBinding> bindings(total_bindings);

    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    for (uint32_t i = 0; i < num_ssbos; i++) {
        bindings[2 + i].binding         = 2 + i;
        bindings[2 + i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2 + i].descriptorCount = 1;
        bindings[2 + i].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                          VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    }

    uint32_t ai_albedo = 2 + num_ssbos;     // array index for albedo entry
    uint32_t ai_normal = 2 + num_ssbos + 1; // array index for normal entry
    uint32_t ai_tex    = 2 + num_ssbos + 2; // array index for texture array entry

    bindings[ai_albedo].binding         = bind_albedo;
    bindings[ai_albedo].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[ai_albedo].descriptorCount = 1;
    bindings[ai_albedo].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    bindings[ai_normal].binding         = bind_normal;
    bindings[ai_normal].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[ai_normal].descriptorCount = 1;
    bindings[ai_normal].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    // Texture array: variable count, partially bound, highest binding number.
    bindings[ai_tex].binding         = bind_tex;
    bindings[ai_tex].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[ai_tex].descriptorCount = MAX_TEX;
    bindings[ai_tex].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    std::vector<VkDescriptorBindingFlags> binding_flags(total_bindings, 0);
    binding_flags[ai_tex] = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT |
                            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

    VkDescriptorSetLayoutBindingFlagsCreateInfo flags_ci{};
    flags_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flags_ci.bindingCount  = total_bindings;
    flags_ci.pBindingFlags = binding_flags.data();

    VkDescriptorSetLayoutCreateInfo layout_ci{};
    layout_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_ci.pNext        = &flags_ci;
    layout_ci.bindingCount = total_bindings;
    layout_ci.pBindings    = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &layout_ci, nullptr, &descriptor_set_layout) != VK_SUCCESS)
        throw std::runtime_error("RT output descriptor set layout creation failed");

    // ------------------------------------------------------------------ Descriptor pool
    std::vector<VkDescriptorPoolSize> pool_sizes;
    pool_sizes.push_back({VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1});
    pool_sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              3}); // colour + albedo + normal
    if (num_ssbos > 0)
        pool_sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, num_ssbos});
    if (num_tex > 0)
        pool_sizes.push_back({VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, num_tex});

    VkDescriptorPoolCreateInfo pool_ci{};
    pool_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets       = 1;
    pool_ci.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_ci.pPoolSizes    = pool_sizes.data();
    if (vkCreateDescriptorPool(device, &pool_ci, nullptr, &descriptor_pool) != VK_SUCCESS)
        throw std::runtime_error("RT output descriptor pool creation failed");

    // ------------------------------------------------------------------ Descriptor set (variable count)
    uint32_t var_count = num_tex;
    VkDescriptorSetVariableDescriptorCountAllocateInfo var_ai{};
    var_ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
    var_ai.descriptorSetCount = 1;
    var_ai.pDescriptorCounts  = &var_count;

    VkDescriptorSetAllocateInfo set_ai{};
    set_ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_ai.pNext              = &var_ai;
    set_ai.descriptorPool     = descriptor_pool;
    set_ai.descriptorSetCount = 1;
    set_ai.pSetLayouts        = &descriptor_set_layout;
    if (vkAllocateDescriptorSets(device, &set_ai, &descriptor_set) != VK_SUCCESS)
        throw std::runtime_error("RT output descriptor set allocation failed");

    create_image(ctx, extent);
    write_descriptors(tlas);
}

RtOutput::~RtOutput() {
    destroy_image();
    vkDestroyDescriptorPool     (device, descriptor_pool,       nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
}

void RtOutput::recreate(VkContext& ctx, VkExtent2D extent, VkAccelerationStructureKHR tlas) {
    destroy_image();
    create_image(ctx, extent);
    write_descriptors(tlas);
}

static VkImage create_storage_image(VkContext& ctx, VkExtent2D extent,
                                    VmaAllocation& out_alloc, VkImageView& out_view) {
    VkImageCreateInfo img_ci{};
    img_ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType     = VK_IMAGE_TYPE_2D;
    img_ci.format        = STORAGE_FORMAT;
    img_ci.extent        = {extent.width, extent.height, 1};
    img_ci.mipLevels     = 1;
    img_ci.arrayLayers   = 1;
    img_ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    img_ci.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo img_ai{};
    img_ai.usage = VMA_MEMORY_USAGE_AUTO;
    img_ai.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VkImage new_image;
    if (vmaCreateImage(ctx.allocator, &img_ci, &img_ai, &new_image, &out_alloc, nullptr) != VK_SUCCESS)
        throw std::runtime_error("RT storage image creation failed");

    VkImageViewCreateInfo view_ci{};
    view_ci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image            = new_image;
    view_ci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format           = STORAGE_FORMAT;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(ctx.device.device, &view_ci, nullptr, &out_view) != VK_SUCCESS)
        throw std::runtime_error("RT storage image view creation failed");

    return new_image;
}

void RtOutput::create_image(VkContext& ctx, VkExtent2D extent) {
    // Accumulation image
    image = create_storage_image(ctx, extent, alloc, view);

    // G-buffer images
    albedo_image = create_storage_image(ctx, extent, albedo_alloc, albedo_view);
    normal_image = create_storage_image(ctx, extent, normal_alloc, normal_view);

    // Transition accumulation and G-buffer images to GENERAL
    VkCommandBuffer cmd = ctx.begin_one_shot();

    auto transition = [&](VkImage img) {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask       = 0;
        b.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.image               = img;
        b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 0, nullptr, 0, nullptr, 1, &b);
    };
    transition(image);
    transition(albedo_image);
    transition(normal_image);

    ctx.end_one_shot(cmd);
}

void RtOutput::destroy_image() {
    // G-buffer images
    if (albedo_view  != VK_NULL_HANDLE) vkDestroyImageView(device, albedo_view, nullptr);
    if (albedo_image != VK_NULL_HANDLE) vmaDestroyImage(allocator, albedo_image, albedo_alloc);
    albedo_view = VK_NULL_HANDLE;  albedo_image = VK_NULL_HANDLE;

    if (normal_view  != VK_NULL_HANDLE) vkDestroyImageView(device, normal_view, nullptr);
    if (normal_image != VK_NULL_HANDLE) vmaDestroyImage(allocator, normal_image, normal_alloc);
    normal_view = VK_NULL_HANDLE;  normal_image = VK_NULL_HANDLE;

    // Accumulation image
    if (view  != VK_NULL_HANDLE) vkDestroyImageView(device, view, nullptr);
    if (image != VK_NULL_HANDLE) vmaDestroyImage(allocator, image, alloc);
    view  = VK_NULL_HANDLE;
    image = VK_NULL_HANDLE;
}

void RtOutput::write_descriptors(VkAccelerationStructureKHR tlas) {
    uint32_t num_ssbos   = static_cast<uint32_t>(ssbos_.size());
    uint32_t bind_albedo = 2 + num_ssbos + 1;
    uint32_t bind_normal = 2 + num_ssbos + 2;
    uint32_t bind_tex    = 2 + num_ssbos + 3;

    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(5 + ssbos_.size());

    // Binding 0: TLAS
    VkWriteDescriptorSetAccelerationStructureKHR as_write{};
    as_write.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    as_write.accelerationStructureCount = 1;
    as_write.pAccelerationStructures    = &tlas;

    VkWriteDescriptorSet w0{};
    w0.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w0.pNext           = &as_write;
    w0.dstSet          = descriptor_set;
    w0.dstBinding      = 0;
    w0.descriptorCount = 1;
    w0.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writes.push_back(w0);

    // Binding 1: colour accumulation storage image
    VkDescriptorImageInfo color_info{};
    color_info.imageView   = view;
    color_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet w1{};
    w1.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w1.dstSet          = descriptor_set;
    w1.dstBinding      = 1;
    w1.descriptorCount = 1;
    w1.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w1.pImageInfo      = &color_info;
    writes.push_back(w1);

    // Bindings 2+: SSBOs
    std::vector<VkDescriptorBufferInfo> buf_infos;
    buf_infos.reserve(ssbos_.size());
    for (const auto& s : ssbos_)
        buf_infos.push_back({s.buf, 0, s.size});

    for (uint32_t i = 0; i < ssbos_.size(); i++) {
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = descriptor_set;
        w.dstBinding      = 2 + i;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo     = &buf_infos[i];
        writes.push_back(w);
    }

    // Binding bind_albedo: albedo G-buffer storage image
    VkDescriptorImageInfo albedo_info{};
    albedo_info.imageView   = albedo_view;
    albedo_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet wa{};
    wa.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wa.dstSet          = descriptor_set;
    wa.dstBinding      = bind_albedo;
    wa.descriptorCount = 1;
    wa.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wa.pImageInfo      = &albedo_info;
    writes.push_back(wa);

    // Binding bind_normal: normal G-buffer storage image
    VkDescriptorImageInfo normal_info{};
    normal_info.imageView   = normal_view;
    normal_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet wn{};
    wn.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wn.dstSet          = descriptor_set;
    wn.dstBinding      = bind_normal;
    wn.descriptorCount = 1;
    wn.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wn.pImageInfo      = &normal_info;
    writes.push_back(wn);

    // Texture array (bind_tex): one entry per texture.
    std::vector<VkDescriptorImageInfo> img_infos;
    if (!tex_views_.empty() && sampler_ != VK_NULL_HANDLE) {
        img_infos.reserve(tex_views_.size());
        for (VkImageView v : tex_views_)
            img_infos.push_back({sampler_, v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});

        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = descriptor_set;
        w.dstBinding      = bind_tex;
        w.descriptorCount = static_cast<uint32_t>(img_infos.size());
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo      = img_infos.data();
        writes.push_back(w);
    }

    vkUpdateDescriptorSets(device,
        static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}
