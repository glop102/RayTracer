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

    uint32_t num_ssbos   = static_cast<uint32_t>(ssbos_.size());
    uint32_t num_tex     = static_cast<uint32_t>(tex_views_.size());
    uint32_t tex_binding = 2 + num_ssbos;
    // max_tex: upper bound declared in the layout (variable count at alloc time).
    // Must be at least 1 so the binding is valid; the actual allocated count can be 0.
    static constexpr uint32_t MAX_TEX = 65536;

    // ------------------------------------------------------------------ Descriptor set layout
    uint32_t total_bindings = tex_binding + 1;  // includes the texture array slot
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

    // Texture array: variable count, partially bound (safe even when num_tex == 0).
    bindings[tex_binding].binding         = tex_binding;
    bindings[tex_binding].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[tex_binding].descriptorCount = MAX_TEX;
    bindings[tex_binding].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    // Flags: only the last binding (texture array) needs VARIABLE | PARTIALLY_BOUND.
    std::vector<VkDescriptorBindingFlags> binding_flags(total_bindings, 0);
    binding_flags[tex_binding] = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT |
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
    pool_sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              1});
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

void RtOutput::create_image(VkContext& ctx, VkExtent2D extent) {
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

    if (vmaCreateImage(ctx.allocator, &img_ci, &img_ai, &image, &alloc, nullptr) != VK_SUCCESS)
        throw std::runtime_error("RT storage image creation failed");

    VkImageViewCreateInfo view_ci{};
    view_ci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image            = image;
    view_ci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format           = STORAGE_FORMAT;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(device, &view_ci, nullptr, &view) != VK_SUCCESS)
        throw std::runtime_error("RT storage image view creation failed");

    VkCommandBuffer cmd = ctx.begin_one_shot();

    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask       = 0;
    barrier.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
    barrier.image               = image;
    barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    ctx.end_one_shot(cmd);
}

void RtOutput::destroy_image() {
    if (view  != VK_NULL_HANDLE) vkDestroyImageView(device, view, nullptr);
    if (image != VK_NULL_HANDLE) vmaDestroyImage(allocator, image, alloc);
    view  = VK_NULL_HANDLE;
    image = VK_NULL_HANDLE;
}

void RtOutput::write_descriptors(VkAccelerationStructureKHR tlas) {
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(2 + ssbos_.size());

    // Binding 0: TLAS
    VkWriteDescriptorSetAccelerationStructureKHR as_write{};
    as_write.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    as_write.accelerationStructureCount = 1;
    as_write.pAccelerationStructures    = &tlas;

    VkWriteDescriptorSet w0{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w0.pNext           = &as_write;
    w0.dstSet          = descriptor_set;
    w0.dstBinding      = 0;
    w0.descriptorCount = 1;
    w0.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writes.push_back(w0);

    // Binding 1: storage image
    VkDescriptorImageInfo img_info{};
    img_info.imageView   = view;
    img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet w1{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w1.dstSet          = descriptor_set;
    w1.dstBinding      = 1;
    w1.descriptorCount = 1;
    w1.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w1.pImageInfo      = &img_info;
    writes.push_back(w1);

    // Bindings 2+: SSBOs — buf_infos must outlive the loop that builds writes.
    std::vector<VkDescriptorBufferInfo> buf_infos;
    buf_infos.reserve(ssbos_.size());
    for (const auto& s : ssbos_)
        buf_infos.push_back({s.buf, 0, s.size});

    for (uint32_t i = 0; i < ssbos_.size(); i++) {
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet          = descriptor_set;
        w.dstBinding      = 2 + i;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo     = &buf_infos[i];
        writes.push_back(w);
    }

    // Texture array (binding 2+N): one entry per texture.
    uint32_t tex_binding = 2 + static_cast<uint32_t>(ssbos_.size());
    std::vector<VkDescriptorImageInfo> img_infos;
    if (!tex_views_.empty() && sampler_ != VK_NULL_HANDLE) {
        img_infos.reserve(tex_views_.size());
        for (VkImageView v : tex_views_)
            img_infos.push_back({sampler_, v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});

        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet          = descriptor_set;
        w.dstBinding      = tex_binding;
        w.descriptorCount = static_cast<uint32_t>(img_infos.size());
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo      = img_infos.data();
        writes.push_back(w);
    }

    vkUpdateDescriptorSets(device,
        static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}
