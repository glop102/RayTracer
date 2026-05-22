#pragma once
#include "gpu_buffer.h"
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>
#include <span>
#include <vector>

struct VkContext;

// Owns the RGBA32F accumulation image and two G-buffer images (albedo, normal),
// plus the descriptor pool/set that binds:
//   binding 0        : TLAS                              (raygen)
//   binding 1        : storage image — colour accum      (raygen)
//   binding 2 .. 2+N : SSBOs from caller                 (raygen | closest-hit)
//   binding 2+N+1    : storage image — albedo G-buffer   (raygen)
//   binding 2+N+2    : storage image — normal G-buffer   (raygen)
//   binding 2+N+3    : COMBINED_IMAGE_SAMPLER array, variable count (closest-hit)
// Must be recreated on swapchain resize; SSBOs and textures are stable across resizes.
struct RtOutput {
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool      descriptor_pool       = VK_NULL_HANDLE;
    VkDescriptorSet       descriptor_set        = VK_NULL_HANDLE;

    // Accumulation image (written + read by raygen every frame)
    VkImage       image = VK_NULL_HANDLE;
    VmaAllocation alloc = {};
    VkImageView   view  = VK_NULL_HANDLE;

    // G-buffer storage images (written by raygen; read by the active denoiser)
    VkImage       albedo_image = VK_NULL_HANDLE;
    VmaAllocation albedo_alloc = {};
    VkImageView   albedo_view  = VK_NULL_HANDLE;

    VkImage       normal_image = VK_NULL_HANDLE;
    VmaAllocation normal_alloc = {};
    VkImageView   normal_view  = VK_NULL_HANDLE;

    RtOutput(VkContext& ctx, VkExtent2D extent, VkAccelerationStructureKHR tlas,
             std::span<const GpuBuffer> ssbos,
             std::span<const VkImageView> tex_views = {},
             VkSampler sampler = VK_NULL_HANDLE);
    ~RtOutput();

    // Recreate images/staging for a new extent and rebind all descriptors.
    void recreate(VkContext& ctx, VkExtent2D extent, VkAccelerationStructureKHR tlas);

    RtOutput(const RtOutput&)            = delete;
    RtOutput& operator=(const RtOutput&) = delete;

private:
    VkDevice     device    = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;

    std::vector<GpuBuffer>   ssbos_;
    std::vector<VkImageView> tex_views_;
    VkSampler                sampler_ = VK_NULL_HANDLE;

    void create_image(VkContext& ctx, VkExtent2D extent);
    void destroy_image();
    void write_descriptors(VkAccelerationStructureKHR tlas);
};
