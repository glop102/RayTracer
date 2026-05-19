#pragma once
#include "gpu_buffer.h"
#include <vulkan/vulkan.h>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>

struct VkContext;

struct Mesh {
    GpuBuffer vertex_buf;
    GpuBuffer index_buf;
    uint32_t  vertex_count = 0;
    uint32_t  index_count  = 0;
    VkDeviceAddress vertex_addr = 0;  // for BLAS geometry input
    VkDeviceAddress index_addr  = 0;

    // CPU-side geometry retained after upload for light triangle extraction.
    std::vector<glm::vec3> cpu_positions;
    std::vector<uint32_t>  cpu_indices;

    Mesh(VkContext& ctx, const std::string& ply_path);
    Mesh(VkContext& ctx, glm::vec3 min_pt, glm::vec3 max_pt); // axis-aligned box (6 faces, face normals)
    // From pre-interleaved {x,y,z,nx,ny,nz,u,v} data — for GLTF-loaded meshes.
    Mesh(VkContext& ctx, std::vector<float> vdata, std::vector<uint32_t> inds);
    ~Mesh();

    Mesh(const Mesh&)            = delete;
    Mesh& operator=(const Mesh&) = delete;

private:
    VkDevice     device    = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;

    void resolve_addresses();
};
