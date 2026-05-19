#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

struct VkContext;

// Per-mesh pointers into device memory — stored in a SSBO and indexed by mesh_index.
// Layout matches the GLSL MeshRef struct (std430).
struct GpuMeshRef {
    VkDeviceAddress vertex_addr;
    VkDeviceAddress index_addr;
};

// Material parameters — stored in a SSBO and indexed by material_index.
// Layout matches the GLSL GpuMaterial struct (std430).
// {vec3, float} packing avoids std430 padding between members.
struct GpuMaterial {
    glm::vec3 diffuse;    float roughness;
    glm::vec3 specular;   float ior;        // index of refraction; 0 = opaque
    glm::vec3 emissive;   float _pad1;
    glm::vec3 absorption; float _pad2;      // Beer-Lambert coefficient (per channel); 0 = no absorption
};

// Per-instance lookup: which mesh and material does each TLAS instance use.
// Indexed by gl_InstanceCustomIndexEXT. Padded to 16 bytes for std430 alignment.
struct GpuInstanceData {
    uint32_t mesh_index;
    uint32_t material_index;
    uint32_t _pad[2];
};

// One emissive triangle in world space — stored in a SSBO for NEE light sampling.
// Matches the GLSL LightTriangle struct (std430, 4×vec4 = 64 bytes).
struct GpuLightTriangle {
    glm::vec3 v0;       float _pad0;
    glm::vec3 v1;       float _pad1;
    glm::vec3 v2;       float _pad2;
    glm::vec3 emission; float _pad3;
};

// Owns the four SSBOs the shaders read to resolve per-hit data and sample lights.
// These never change after construction (static scene description).
struct SceneData {
    VkBuffer     mesh_refs_buf        = VK_NULL_HANDLE;
    VkDeviceSize mesh_refs_range      = 0;
    VkBuffer     materials_buf        = VK_NULL_HANDLE;
    VkDeviceSize materials_range      = 0;
    VkBuffer     instances_buf        = VK_NULL_HANDLE;
    VkDeviceSize instances_range      = 0;
    VkBuffer     light_triangles_buf  = VK_NULL_HANDLE;
    VkDeviceSize light_triangles_range= 0;
    uint32_t     light_count          = 0;

    SceneData(VkContext& ctx,
              const std::vector<GpuMeshRef>&        mesh_refs,
              const std::vector<GpuMaterial>&       materials,
              const std::vector<GpuInstanceData>&   instances,
              const std::vector<GpuLightTriangle>&  light_triangles);
    ~SceneData();

    SceneData(const SceneData&)            = delete;
    SceneData& operator=(const SceneData&) = delete;

private:
    VkDevice     device    = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;

    VmaAllocation mesh_refs_alloc       = {};
    VmaAllocation materials_alloc       = {};
    VmaAllocation instances_alloc       = {};
    VmaAllocation light_triangles_alloc = {};
};
