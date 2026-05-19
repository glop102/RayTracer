#pragma once
#include "gpu_buffer.h"
#include "mesh.h"

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
// Layout matches the GLSL GpuMaterial struct (std430, 80 bytes = 5×vec4).
// {vec3, float} packing avoids std430 padding between members.
struct GpuMaterial {
    glm::vec3 diffuse;    float roughness;
    glm::vec3 specular;   float ior;        // index of refraction; 0 = opaque
    glm::vec3 emissive;   float metallic;   // PBR metallic factor (0 for non-GLTF materials)
    glm::vec3 absorption; float _pad2;      // Beer-Lambert coefficient (per channel)
    int diffuse_tex;   int mr_tex; int normal_tex; int _pad3;  // -1 = no texture
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

// Resolved per-instance data used for light extraction (and future scene loaders).
struct SceneInstance {
    glm::mat4 transform;
    uint32_t  mesh_index;
    uint32_t  material_index;
};

// Scan all instances for non-zero emissive materials and collect their
// world-space triangles into a NEE light list. Back-facing triangles are
// included — the shader's cos_theta_l check discards them without bias.
std::vector<GpuLightTriangle> extract_light_triangles(
    const std::vector<SceneInstance>& instances,
    const std::vector<const Mesh*>&   meshes,
    const std::vector<GpuMaterial>&   materials);

// Owns the SSBOs the shaders read to resolve per-hit data and sample lights.
struct SceneData {
    GpuBuffer mesh_refs;
    GpuBuffer materials;
    GpuBuffer instances;
    GpuBuffer light_triangles;
    uint32_t  light_count = 0;

    SceneData(VkContext& ctx,
              const std::vector<GpuMeshRef>&       mesh_refs,
              const std::vector<GpuMaterial>&      materials,
              const std::vector<GpuInstanceData>&  instances,
              const std::vector<GpuLightTriangle>& light_triangles);
    ~SceneData();

    SceneData(const SceneData&)            = delete;
    SceneData& operator=(const SceneData&) = delete;

private:
    VmaAllocator allocator = VK_NULL_HANDLE;
};
