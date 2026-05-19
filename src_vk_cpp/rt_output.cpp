#include "rt_output.h"
#include "scene_data.h"
#include "vk_context.h"

#include <stdexcept>

// RGBA32F for HDR path-trace accumulation (running mean, linear space).
// The blit to the SRGB swapchain applies the sRGB transfer function.
static constexpr VkFormat STORAGE_FORMAT = VK_FORMAT_R32G32B32A32_SFLOAT;

RtOutput::RtOutput(VkContext& ctx, VkExtent2D extent, VkAccelerationStructureKHR tlas,
                   const SceneData& scene) {
    device         = ctx.device.device;
    allocator      = ctx.allocator;
    mesh_refs_buf        = scene.mesh_refs_buf;
    mesh_refs_range      = scene.mesh_refs_range;
    materials_buf        = scene.materials_buf;
    materials_range      = scene.materials_range;
    instances_buf        = scene.instances_buf;
    instances_range      = scene.instances_range;
    light_triangles_buf  = scene.light_triangles_buf;
    light_triangles_range= scene.light_triangles_range;

    // ------------------------------------------------------------------ Descriptor set layout
    // binding 0: TLAS                  — raygen
    // binding 1: storage image         — raygen (RGBA32F running-mean accumulation)
    // binding 2: mesh_refs SSBO        — closest-hit
    // binding 3: materials SSBO        — closest-hit
    // binding 4: instances SSBO        — closest-hit
    // binding 5: light_triangles SSBO  — raygen (NEE light sampling)
    VkDescriptorSetLayoutBinding bindings[6]{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    bindings[2].binding         = 2;
    bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    bindings[3].binding         = 3;
    bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    bindings[4].binding         = 4;
    bindings[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags      = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    bindings[5].binding         = 5;
    bindings[5].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    VkDescriptorSetLayoutCreateInfo layout_ci{};
    layout_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_ci.bindingCount = 6;
    layout_ci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(device, &layout_ci, nullptr, &descriptor_set_layout) != VK_SUCCESS)
        throw std::runtime_error("RT output descriptor set layout creation failed");

    // ------------------------------------------------------------------ Descriptor pool
    VkDescriptorPoolSize pool_sizes[3]{};
    pool_sizes[0].type            = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    pool_sizes[0].descriptorCount = 1;
    pool_sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[1].descriptorCount = 1;
    pool_sizes[2].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[2].descriptorCount = 4;  // mesh_refs + materials + instances + light_triangles

    VkDescriptorPoolCreateInfo pool_ci{};
    pool_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets       = 1;
    pool_ci.poolSizeCount = 3;
    pool_ci.pPoolSizes    = pool_sizes;
    if (vkCreateDescriptorPool(device, &pool_ci, nullptr, &descriptor_pool) != VK_SUCCESS)
        throw std::runtime_error("RT output descriptor pool creation failed");

    // ------------------------------------------------------------------ Descriptor set
    VkDescriptorSetAllocateInfo set_ai{};
    set_ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
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

    // Transition from UNDEFINED to GENERAL so the raygen shader can write.
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
    // Binding 0: TLAS
    VkWriteDescriptorSetAccelerationStructureKHR as_write{};
    as_write.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    as_write.accelerationStructureCount = 1;
    as_write.pAccelerationStructures    = &tlas;

    VkWriteDescriptorSet write0{};
    write0.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write0.pNext           = &as_write;
    write0.dstSet          = descriptor_set;
    write0.dstBinding      = 0;
    write0.descriptorCount = 1;
    write0.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    // Binding 1: storage image
    VkDescriptorImageInfo img_info{};
    img_info.imageView   = view;
    img_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write1{};
    write1.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write1.dstSet          = descriptor_set;
    write1.dstBinding      = 1;
    write1.descriptorCount = 1;
    write1.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write1.pImageInfo      = &img_info;

    // Binding 2: mesh_refs SSBO
    VkDescriptorBufferInfo mesh_refs_info{mesh_refs_buf, 0, mesh_refs_range};
    VkWriteDescriptorSet write2{};
    write2.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write2.dstSet          = descriptor_set;
    write2.dstBinding      = 2;
    write2.descriptorCount = 1;
    write2.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write2.pBufferInfo     = &mesh_refs_info;

    // Binding 3: materials SSBO
    VkDescriptorBufferInfo materials_info{materials_buf, 0, materials_range};
    VkWriteDescriptorSet write3{};
    write3.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write3.dstSet          = descriptor_set;
    write3.dstBinding      = 3;
    write3.descriptorCount = 1;
    write3.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write3.pBufferInfo     = &materials_info;

    // Binding 4: instances SSBO
    VkDescriptorBufferInfo instances_info{instances_buf, 0, instances_range};
    VkWriteDescriptorSet write4{};
    write4.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write4.dstSet          = descriptor_set;
    write4.dstBinding      = 4;
    write4.descriptorCount = 1;
    write4.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write4.pBufferInfo     = &instances_info;

    // Binding 5: light_triangles SSBO
    VkDescriptorBufferInfo light_triangles_info{light_triangles_buf, 0, light_triangles_range};
    VkWriteDescriptorSet write5{};
    write5.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write5.dstSet          = descriptor_set;
    write5.dstBinding      = 5;
    write5.descriptorCount = 1;
    write5.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write5.pBufferInfo     = &light_triangles_info;

    VkWriteDescriptorSet writes[6] = {write0, write1, write2, write3, write4, write5};
    vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);
}
