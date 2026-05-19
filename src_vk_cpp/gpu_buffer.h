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

// Upload data via a staging buffer into a device-local buffer with the given usage flags.
// Blocks until the transfer is complete (vkQueueWaitIdle inside end_one_shot).
GpuBuffer upload_device_buffer(VkContext& ctx, VkBufferUsageFlags usage,
                                const void* data, VkDeviceSize size);

// Return the device address of a buffer (requires VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT).
VkDeviceAddress buffer_device_address(VkDevice dev, VkBuffer buf);

// Release the buffer memory and zero the struct.
void destroy_buffer(VmaAllocator allocator, GpuBuffer& buf);
