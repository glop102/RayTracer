#pragma once
#include "mesh.h"
#include "gpu_texture.h"
#include "scene_data.h"
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

struct VkContext;

struct GltfInstance {
    uint32_t  mesh_index;
    uint32_t  material_index;
    glm::mat4 transform;
    uint32_t  sbt_record_offset = 0;  // 0 = opaque
};

// All GPU-resident data for a loaded GLTF scene.
// Call destroy() before the VkContext is torn down.
struct LoadedScene {
    std::vector<std::unique_ptr<Mesh>> meshes;
    std::vector<bool>                  mesh_is_glass;  // parallel to meshes; true → non-opaque BLAS
    std::vector<GpuMaterial>           materials;
    std::vector<GltfInstance>          instances;
    std::vector<GpuTexture>            textures;
    VkSampler                          sampler = VK_NULL_HANDLE;

    void destroy(VkDevice device, VmaAllocator allocator);
};

// Load a .gltf or .glb file. Uploads all meshes and textures to the GPU.
LoadedScene load_gltf(VkContext& ctx, const std::string& path);
