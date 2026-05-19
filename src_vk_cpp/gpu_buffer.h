#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

struct VkContext;

// Handle + allocation + size bundled so they always travel together.
struct GpuBuffer {
    VkBuffer      buf   = VK_NULL_HANDLE;
    VmaAllocation alloc = {};
    VkDeviceSize  size  = 0;
};

// Upload arbitrary data as a persistently-mapped host-visible SSBO.
GpuBuffer upload_ssbo(VkContext& ctx, const void* data, VkDeviceSize size);

// Release the buffer memory and zero the struct.
void destroy_buffer(VmaAllocator allocator, GpuBuffer& buf);
