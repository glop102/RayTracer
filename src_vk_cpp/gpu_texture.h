#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>
#include <cstdint>

struct VkContext;

// Plain-data handle for a 2D device-local texture image.
// Destroy with destroy_texture(); no RAII.
struct GpuTexture {
    VkImage       image = VK_NULL_HANDLE;
    VmaAllocation alloc = {};
    VkImageView   view  = VK_NULL_HANDLE;
};

// Upload RGBA8 pixel data into a SHADER_READ_ONLY_OPTIMAL device-local image.
// fmt: VK_FORMAT_R8G8B8A8_SRGB for colour textures, UNORM for linear data.
GpuTexture upload_texture(VkContext& ctx, const uint8_t* rgba,
                          uint32_t width, uint32_t height,
                          VkFormat fmt = VK_FORMAT_R8G8B8A8_SRGB);

void destroy_texture(VkDevice device, VmaAllocator allocator, GpuTexture& tex);
