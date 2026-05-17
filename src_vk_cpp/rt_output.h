#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

struct VkContext;
struct SceneData;

// Owns the RGBA32F storage image for path-trace accumulation, plus the
// descriptor pool and set that binds:
//   binding 0: TLAS
//   binding 1: storage image (RGBA32F, read-modify-write for running mean)
//   binding 2: vertex SSBO  (for face normal lookup in closest-hit)
//   binding 3: index  SSBO
// Must be recreated on swapchain resize; vertex/index buffers are stable.
struct RtOutput {
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool      descriptor_pool       = VK_NULL_HANDLE;
    VkDescriptorSet       descriptor_set        = VK_NULL_HANDLE;

    VkImage       image = VK_NULL_HANDLE;
    VmaAllocation alloc = {};
    VkImageView   view  = VK_NULL_HANDLE;

    RtOutput(VkContext& ctx, VkExtent2D extent, VkAccelerationStructureKHR tlas,
             const SceneData& scene);
    ~RtOutput();

    // Recreate the storage image for a new extent and rebind all descriptors.
    void recreate(VkContext& ctx, VkExtent2D extent, VkAccelerationStructureKHR tlas);

    RtOutput(const RtOutput&)            = delete;
    RtOutput& operator=(const RtOutput&) = delete;

private:
    VkDevice     device    = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;

    VkBuffer     mesh_refs_buf   = VK_NULL_HANDLE;
    VkDeviceSize mesh_refs_range = 0;
    VkBuffer     materials_buf   = VK_NULL_HANDLE;
    VkDeviceSize materials_range = 0;
    VkBuffer     instances_buf   = VK_NULL_HANDLE;
    VkDeviceSize instances_range = 0;

    void create_image(VkContext& ctx, VkExtent2D extent);
    void destroy_image();
    void write_descriptors(VkAccelerationStructureKHR tlas);
};
