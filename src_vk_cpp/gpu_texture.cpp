#include "gpu_texture.h"
#include "gpu_buffer.h"
#include "vk_context.h"

#include <cstring>
#include <stdexcept>

GpuTexture upload_texture(VkContext& ctx, const uint8_t* rgba,
                          uint32_t width, uint32_t height, VkFormat fmt) {
    VkDeviceSize byte_size = (VkDeviceSize)width * height * 4;

    // Host-visible staging buffer
    VkBufferCreateInfo stg_ci{};
    stg_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stg_ci.size  = byte_size;
    stg_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stg_ai{};
    stg_ai.usage = VMA_MEMORY_USAGE_AUTO;
    stg_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                   VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer stg_buf; VmaAllocation stg_alloc; VmaAllocationInfo stg_info{};
    if (vmaCreateBuffer(ctx.allocator, &stg_ci, &stg_ai,
                        &stg_buf, &stg_alloc, &stg_info) != VK_SUCCESS)
        throw std::runtime_error("Texture staging buffer creation failed");
    std::memcpy(stg_info.pMappedData, rgba, byte_size);
    vmaFlushAllocation(ctx.allocator, stg_alloc, 0, VK_WHOLE_SIZE);

    // Device-local image
    GpuTexture tex{};
    VkImageCreateInfo img_ci{};
    img_ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType     = VK_IMAGE_TYPE_2D;
    img_ci.format        = fmt;
    img_ci.extent        = {width, height, 1};
    img_ci.mipLevels     = 1;
    img_ci.arrayLayers   = 1;
    img_ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    img_ci.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo img_ai{};
    img_ai.usage = VMA_MEMORY_USAGE_AUTO;
    img_ai.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    if (vmaCreateImage(ctx.allocator, &img_ci, &img_ai,
                       &tex.image, &tex.alloc, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Texture image creation failed");

    // UNDEFINED → TRANSFER_DST, copy, TRANSFER_DST → SHADER_READ_ONLY
    VkCommandBuffer cmd = ctx.begin_one_shot();

    auto barrier = [&](VkImageLayout old_l, VkImageLayout new_l,
                       VkAccessFlags src_a, VkAccessFlags dst_a,
                       VkPipelineStageFlags src_s, VkPipelineStageFlags dst_s) {
        VkImageMemoryBarrier b{};
        b.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask    = src_a;
        b.dstAccessMask    = dst_a;
        b.oldLayout        = old_l;
        b.newLayout        = new_l;
        b.image            = tex.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, src_s, dst_s, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {width, height, 1};
    vkCmdCopyBufferToImage(cmd, stg_buf, tex.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    ctx.end_one_shot(cmd);
    vmaDestroyBuffer(ctx.allocator, stg_buf, stg_alloc);

    VkImageViewCreateInfo view_ci{};
    view_ci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image            = tex.image;
    view_ci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format           = fmt;
    view_ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(ctx.device.device, &view_ci, nullptr, &tex.view) != VK_SUCCESS)
        throw std::runtime_error("Texture image view creation failed");

    return tex;
}

void destroy_texture(VkDevice device, VmaAllocator allocator, GpuTexture& tex) {
    if (tex.view  != VK_NULL_HANDLE) vkDestroyImageView(device, tex.view, nullptr);
    if (tex.image != VK_NULL_HANDLE) vmaDestroyImage(allocator, tex.image, tex.alloc);
    tex = {};
}
