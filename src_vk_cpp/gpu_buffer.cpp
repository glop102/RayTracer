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

void destroy_buffer(VmaAllocator allocator, GpuBuffer& buf) {
    if (buf.buf != VK_NULL_HANDLE)
        vmaDestroyBuffer(allocator, buf.buf, buf.alloc);
    buf = {};
}
