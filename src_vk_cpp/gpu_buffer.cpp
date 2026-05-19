#include "gpu_buffer.h"
#include "vk_context.h"

#include <cstring>
#include <stdexcept>

GpuBuffer upload_ssbo(VkContext& ctx, const void* data, VkDeviceSize size) {
    GpuBuffer gb;
    gb.size = size;

    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size  = size;
    ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
               VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info;
    if (vmaCreateBuffer(ctx.allocator, &ci, &ai, &gb.buf, &gb.alloc, &info) != VK_SUCCESS)
        throw std::runtime_error("SSBO upload failed");
    std::memcpy(info.pMappedData, data, size);
    return gb;
}

GpuBuffer upload_device_buffer(VkContext& ctx, VkBufferUsageFlags usage,
                                const void* data, VkDeviceSize size) {
    // CPU-visible staging buffer
    VkBufferCreateInfo stg_ci{};
    stg_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stg_ci.size  = size;
    stg_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stg_ai{};
    stg_ai.usage = VMA_MEMORY_USAGE_AUTO;
    stg_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VkBuffer stg_buf; VmaAllocation stg_alloc;
    if (vmaCreateBuffer(ctx.allocator, &stg_ci, &stg_ai, &stg_buf, &stg_alloc, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Staging buffer creation failed");

    void* mapped;
    vmaMapMemory(ctx.allocator, stg_alloc, &mapped);
    std::memcpy(mapped, data, size);
    vmaUnmapMemory(ctx.allocator, stg_alloc);

    // Device-local destination buffer
    GpuBuffer gb;
    gb.size = size;

    VkBufferCreateInfo buf_ci{};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size  = size;
    buf_ci.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo buf_ai{};
    buf_ai.usage = VMA_MEMORY_USAGE_AUTO;
    buf_ai.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    if (vmaCreateBuffer(ctx.allocator, &buf_ci, &buf_ai, &gb.buf, &gb.alloc, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Device buffer creation failed");

    VkCommandBuffer cmd = ctx.begin_one_shot();
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cmd, stg_buf, gb.buf, 1, &region);
    ctx.end_one_shot(cmd);  // submits and vkQueueWaitIdle — data fully visible on return

    vmaDestroyBuffer(ctx.allocator, stg_buf, stg_alloc);
    return gb;
}

VkDeviceAddress buffer_device_address(VkDevice dev, VkBuffer buf) {
    VkBufferDeviceAddressInfo info{};
    info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buf;
    return vkGetBufferDeviceAddress(dev, &info);
}

void destroy_buffer(VmaAllocator allocator, GpuBuffer& buf) {
    if (buf.buf != VK_NULL_HANDLE)
        vmaDestroyBuffer(allocator, buf.buf, buf.alloc);
    buf = {};
}
