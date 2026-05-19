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
//   binding 2+N+1    : COMBINED_IMAGE_SAMPLER array, variable count (closest-hit)
// Must be recreated on swapchain resize; SSBOs and textures are stable across resizes.
struct RtOutput {
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool      descriptor_pool       = VK_NULL_HANDLE;
    VkDescriptorSet       descriptor_set        = VK_NULL_HANDLE;

    VkImage       image = VK_NULL_HANDLE;
    VmaAllocation alloc = {};
    VkImageView   view  = VK_NULL_HANDLE;

    RtOutput(VkContext& ctx, VkExtent2D extent, VkAccelerationStructureKHR tlas,
             std::span<const GpuBuffer> ssbos,
             std::span<const VkImageView> tex_views = {},
             VkSampler sampler = VK_NULL_HANDLE);
    ~RtOutput();

    // Recreate the storage image for a new extent and rebind all descriptors.
    void recreate(VkContext& ctx, VkExtent2D extent, VkAccelerationStructureKHR tlas);

    RtOutput(const RtOutput&)            = delete;
    RtOutput& operator=(const RtOutput&) = delete;

private:
    VkDevice     device    = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;

    std::vector<GpuBuffer>   ssbos_;       // non-owning; SceneData owns memory
    std::vector<VkImageView> tex_views_;   // non-owning; LoadedScene owns images
    VkSampler                sampler_ = VK_NULL_HANDLE;  // non-owning

    void create_image(VkContext& ctx, VkExtent2D extent);
    void destroy_image();
    void write_descriptors(VkAccelerationStructureKHR tlas);
};
