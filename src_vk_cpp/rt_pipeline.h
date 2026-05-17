#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

struct VkContext;

// Owns the ray tracing pipeline, its layout, and the shader binding table.
// The descriptor set layout for TLAS + storage image lives in RtOutput.
struct RtPipeline {
    VkPipelineLayout layout   = VK_NULL_HANDLE;
    VkPipeline       pipeline = VK_NULL_HANDLE;

    VkBuffer      sbt_buf   = VK_NULL_HANDLE;
    VmaAllocation sbt_alloc = {};

    VkStridedDeviceAddressRegionKHR raygen_region   = {};
    VkStridedDeviceAddressRegionKHR miss_region     = {};
    VkStridedDeviceAddressRegionKHR hit_region      = {};
    VkStridedDeviceAddressRegionKHR callable_region = {};  // unused, must be zero-initialized

    // descriptor_set_layout comes from RtOutput (TLAS + storage image).
    RtPipeline(VkContext& ctx, VkDescriptorSetLayout rt_output_layout);
    ~RtPipeline();

    RtPipeline(const RtPipeline&)            = delete;
    RtPipeline& operator=(const RtPipeline&) = delete;

private:
    VkDevice     device    = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
};
