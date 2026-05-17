#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>

struct VkContext;

struct Mesh {
    VkBuffer      vertex_buf   = VK_NULL_HANDLE;
    VkBuffer      index_buf    = VK_NULL_HANDLE;
    VmaAllocation vertex_alloc = {};
    VmaAllocation index_alloc  = {};
    uint32_t      index_count  = 0;

    Mesh(VkContext& ctx, const std::string& ply_path);
    ~Mesh();

    Mesh(const Mesh&)            = delete;
    Mesh& operator=(const Mesh&) = delete;

private:
    VkDevice     device    = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
};
