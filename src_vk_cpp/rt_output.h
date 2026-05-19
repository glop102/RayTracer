#pragma once
#include "gpu_buffer.h"
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>
#include <span>
#include <vector>

struct VkContext;

// Owns the RGBA32F accumulation image plus the descriptor pool/set that binds:
//   binding 0        : TLAS               (raygen)
//   binding 1        : storage image      (raygen)
//   binding 2 .. 2+N : SSBOs from caller  (raygen | closest-hit)
// Must be recreated on swapchain resize; SSBOs are stable across resizes.
struct RtOutput {
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool      descriptor_pool       = VK_NULL_HANDLE;
    VkDescriptorSet       descriptor_set        = VK_NULL_HANDLE;

    VkImage       image = VK_NULL_HANDLE;
    VmaAllocation alloc = {};
    VkImageView   view  = VK_NULL_HANDLE;

    RtOutput(VkContext& ctx, VkExtent2D extent, VkAccelerationStructureKHR tlas,
             std::span<const GpuBuffer> ssbos);
    ~RtOutput();

    // Recreate the storage image for a new extent and rebind all descriptors.
    void recreate(VkContext& ctx, VkExtent2D extent, VkAccelerationStructureKHR tlas);

    RtOutput(const RtOutput&)            = delete;
    RtOutput& operator=(const RtOutput&) = delete;

private:
    VkDevice     device    = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;

    std::vector<GpuBuffer> ssbos_;  // non-owning handles; SceneData owns the memory

    void create_image(VkContext& ctx, VkExtent2D extent);
    void destroy_image();
    void write_descriptors(VkAccelerationStructureKHR tlas);
};
