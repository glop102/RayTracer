// TINYGLTF_IMPLEMENTATION must appear in exactly one TU.
// The nix packages for nlohmann_json and stb put headers under include/nlohmann/ and
// include/stb/ respectively, but tiny_gltf.h includes them with bare names.  Pre-include
// via the real paths and suppress tinygltf's own attempts.
#include <nlohmann/json.hpp>
#define TINYGLTF_NO_INCLUDE_JSON
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

#include "gltf_scene.h"
#include "vk_context.h"
#include "gpu_texture.h"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Accessor helpers

static std::vector<float> extract_floats(const tinygltf::Model& m, int acc_idx) {
    const auto& acc = m.accessors[acc_idx];
    const auto& bv  = m.bufferViews[acc.bufferView];
    const auto& buf = m.buffers[bv.buffer];
    int num_comp    = tinygltf::GetNumComponentsInType(acc.type);
    int stride      = acc.ByteStride(bv);

    std::vector<float> out;
    out.reserve(acc.count * num_comp);
    const uint8_t* src = buf.data.data() + bv.byteOffset + acc.byteOffset;
    for (size_t i = 0; i < acc.count; i++) {
        const float* e = reinterpret_cast<const float*>(src + i * stride);
        for (int c = 0; c < num_comp; c++) out.push_back(e[c]);
    }
    return out;
}

static std::vector<uint32_t> extract_indices(const tinygltf::Model& m, int acc_idx) {
    const auto& acc = m.accessors[acc_idx];
    const auto& bv  = m.bufferViews[acc.bufferView];
    const auto& buf = m.buffers[bv.buffer];
    const uint8_t* src = buf.data.data() + bv.byteOffset + acc.byteOffset;

    std::vector<uint32_t> out;
    out.reserve(acc.count);
    if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
        for (size_t i = 0; i < acc.count; i++)
            out.push_back(reinterpret_cast<const uint16_t*>(src)[i]);
    } else if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
        for (size_t i = 0; i < acc.count; i++)
            out.push_back(reinterpret_cast<const uint32_t*>(src)[i]);
    } else if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
        for (size_t i = 0; i < acc.count; i++)
            out.push_back(src[i]);
    }
    return out;
}

// ---------------------------------------------------------------------------

void LoadedScene::destroy(VkDevice device, VmaAllocator allocator) {
    for (auto& t : textures) destroy_texture(device, allocator, t);
    if (sampler != VK_NULL_HANDLE) vkDestroySampler(device, sampler, nullptr);
    sampler = VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------

LoadedScene load_gltf(VkContext& ctx, const std::string& path) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model    model;
    std::string        err, warn;

    bool ok = path.ends_with(".glb")
        ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
        : loader.LoadASCIIFromFile (&model, &err, &warn, path);
    if (!ok) throw std::runtime_error("GLTF load failed: " + err);

    LoadedScene scene;

    // ------------------------------------------------------------------ Images
    // Decide which images are sRGB (base colour / emissive) vs. linear (MR, normals).
    std::vector<bool> is_srgb(model.images.size(), false);
    for (const auto& mat : model.materials) {
        auto mark_srgb = [&](int tex_idx) {
            if (tex_idx >= 0 && model.textures[tex_idx].source >= 0)
                is_srgb[model.textures[tex_idx].source] = true;
        };
        mark_srgb(mat.pbrMetallicRoughness.baseColorTexture.index);
        mark_srgb(mat.emissiveTexture.index);
    }

    std::vector<int> image_to_gpu(model.images.size(), -1);
    for (size_t i = 0; i < model.images.size(); i++) {
        const auto& img = model.images[i];
        if (img.image.empty() || img.width <= 0 || img.height <= 0) continue;

        std::vector<uint8_t> rgba;
        if (img.component == 4) {
            rgba = img.image;
        } else if (img.component == 3) {
            rgba.resize((size_t)img.width * img.height * 4);
            for (int p = 0; p < img.width * img.height; p++) {
                rgba[p*4+0] = img.image[p*3+0];
                rgba[p*4+1] = img.image[p*3+1];
                rgba[p*4+2] = img.image[p*3+2];
                rgba[p*4+3] = 255;
            }
        } else {
            continue;
        }

        VkFormat fmt = is_srgb[i] ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
        image_to_gpu[i] = (int)scene.textures.size();
        scene.textures.push_back(
            upload_texture(ctx, rgba.data(), (uint32_t)img.width, (uint32_t)img.height, fmt));
    }

    auto gltf_tex_to_gpu = [&](int tex_idx) -> int {
        if (tex_idx < 0) return -1;
        int src = model.textures[tex_idx].source;
        if (src < 0 || src >= (int)image_to_gpu.size()) return -1;
        return image_to_gpu[src];
    };

    // ------------------------------------------------------------------ Materials
    for (const auto& mat : model.materials) {
        const auto& pbr   = mat.pbrMetallicRoughness;
        glm::vec3 base_col = {(float)pbr.baseColorFactor[0],
                              (float)pbr.baseColorFactor[1],
                              (float)pbr.baseColorFactor[2]};
        float roughness = (float)pbr.roughnessFactor;
        float metallic  = (float)pbr.metallicFactor;
        glm::vec3 emissive = {(float)mat.emissiveFactor[0],
                              (float)mat.emissiveFactor[1],
                              (float)mat.emissiveFactor[2]};

        // Pre-compute PBR F0 = mix(0.04, baseColor, metallic)
        glm::vec3 F0 = glm::mix(glm::vec3(0.04f), base_col, metallic);

        GpuMaterial gm{};
        gm.diffuse    = base_col;            // base colour factor; shader multiplies by (1-metallic)
        gm.roughness  = roughness;
        gm.specular   = F0;                  // pre-computed for the no-MR-texture path
        gm.ior        = 0.0f;
        gm.emissive   = emissive;
        gm.metallic   = metallic;
        gm.absorption = glm::vec3(0.0f);
        gm._pad2      = 0.0f;
        gm.diffuse_tex   = gltf_tex_to_gpu(pbr.baseColorTexture.index);
        gm.mr_tex        = gltf_tex_to_gpu(pbr.metallicRoughnessTexture.index);
        gm.normal_tex    = gltf_tex_to_gpu(mat.normalTexture.index);
        gm.emissive_tex  = gltf_tex_to_gpu(mat.emissiveTexture.index);
        scene.materials.push_back(gm);
    }
    if (scene.materials.empty()) {
        GpuMaterial def{};
        def.diffuse   = {0.8f, 0.8f, 0.8f};
        def.roughness = 0.5f;
        def.specular  = {0.04f, 0.04f, 0.04f};
        def.diffuse_tex = def.mr_tex = def.normal_tex = def.emissive_tex = -1;
        scene.materials.push_back(def);
    }

    // ------------------------------------------------------------------ Meshes
    // One Mesh per GLTF mesh × primitive; track by (mesh_idx, prim_idx) key.
    std::map<std::pair<int,int>, uint32_t> prim_to_gpu;

    for (int mi = 0; mi < (int)model.meshes.size(); mi++) {
        const auto& gm = model.meshes[mi];
        for (int pi = 0; pi < (int)gm.primitives.size(); pi++) {
            const auto& prim = gm.primitives[pi];
            if (prim.mode != TINYGLTF_MODE_TRIANGLES) continue;

            auto pos_it = prim.attributes.find("POSITION");
            if (pos_it == prim.attributes.end()) continue;
            std::vector<float> pos  = extract_floats(model, pos_it->second);
            uint32_t nv = (uint32_t)(pos.size() / 3);

            std::vector<float> nrm, uv;
            auto norm_it = prim.attributes.find("NORMAL");
            if (norm_it != prim.attributes.end())
                nrm = extract_floats(model, norm_it->second);
            auto uv_it = prim.attributes.find("TEXCOORD_0");
            if (uv_it != prim.attributes.end())
                uv = extract_floats(model, uv_it->second);

            std::vector<uint32_t> inds;
            if (prim.indices >= 0)
                inds = extract_indices(model, prim.indices);
            else {
                inds.reserve(nv);
                for (uint32_t i = 0; i < nv; i++) inds.push_back(i);
            }

            // Generate smooth normals if absent
            if (nrm.empty()) {
                nrm.assign(nv * 3, 0.0f);
                for (size_t t = 0; t + 2 < inds.size(); t += 3) {
                    uint32_t i0 = inds[t], i1 = inds[t+1], i2 = inds[t+2];
                    glm::vec3 p0 = {pos[i0*3], pos[i0*3+1], pos[i0*3+2]};
                    glm::vec3 p1 = {pos[i1*3], pos[i1*3+1], pos[i1*3+2]};
                    glm::vec3 p2 = {pos[i2*3], pos[i2*3+1], pos[i2*3+2]};
                    glm::vec3 fn = glm::cross(p1-p0, p2-p0);
                    for (int c = 0; c < 3; c++) {
                        nrm[i0*3+c] += fn[c]; nrm[i1*3+c] += fn[c]; nrm[i2*3+c] += fn[c];
                    }
                }
                for (uint32_t i = 0; i < nv; i++) {
                    glm::vec3 n = {nrm[i*3], nrm[i*3+1], nrm[i*3+2]};
                    float len = glm::length(n);
                    if (len > 1e-6f) { nrm[i*3] = n.x/len; nrm[i*3+1] = n.y/len; nrm[i*3+2] = n.z/len; }
                }
            }

            // Interleave {x,y,z, nx,ny,nz, u,v}
            std::vector<float> vdata;
            vdata.reserve(nv * 8);
            for (uint32_t i = 0; i < nv; i++) {
                vdata.push_back(pos[i*3]);   vdata.push_back(pos[i*3+1]);   vdata.push_back(pos[i*3+2]);
                vdata.push_back(nrm[i*3]);   vdata.push_back(nrm[i*3+1]);   vdata.push_back(nrm[i*3+2]);
                vdata.push_back(uv.size() >= (i+1)*2 ? uv[i*2]   : 0.0f);
                vdata.push_back(uv.size() >= (i+1)*2 ? uv[i*2+1] : 0.0f);
            }

            uint32_t gpu_idx = (uint32_t)scene.meshes.size();
            prim_to_gpu[{mi, pi}] = gpu_idx;
            scene.meshes.push_back(
                std::make_unique<Mesh>(ctx, std::move(vdata), std::move(inds)));
        }
    }

    // ------------------------------------------------------------------ Nodes
    if (model.scenes.empty()) throw std::runtime_error("GLTF has no scenes");
    const auto& gltf_scene = model.scenes[model.defaultScene >= 0 ? model.defaultScene : 0];

    std::function<void(int, const glm::mat4&)> traverse = [&](int ni, const glm::mat4& parent) {
        const auto& node = model.nodes[ni];
        glm::mat4 local(1.0f);

        if (node.matrix.size() == 16) {
            // column-major, directly usable by GLM
            for (int c = 0; c < 4; c++)
                for (int r = 0; r < 4; r++)
                    local[c][r] = (float)node.matrix[c*4+r];
        } else {
            glm::mat4 T(1), R(1), S(1);
            if (node.translation.size() == 3)
                T = glm::translate(glm::mat4(1.0f),
                    glm::vec3((float)node.translation[0],
                              (float)node.translation[1],
                              (float)node.translation[2]));
            if (node.rotation.size() == 4) {
                // GLTF: [x,y,z,w]
                glm::quat q((float)node.rotation[3],
                            (float)node.rotation[0],
                            (float)node.rotation[1],
                            (float)node.rotation[2]);
                R = glm::mat4_cast(q);
            }
            if (node.scale.size() == 3)
                S = glm::scale(glm::mat4(1.0f),
                    glm::vec3((float)node.scale[0],
                              (float)node.scale[1],
                              (float)node.scale[2]));
            local = T * R * S;
        }

        glm::mat4 world = parent * local;

        if (node.mesh >= 0 && node.mesh < (int)model.meshes.size()) {
            const auto& gm = model.meshes[node.mesh];
            for (int pi = 0; pi < (int)gm.primitives.size(); pi++) {
                auto it = prim_to_gpu.find({node.mesh, pi});
                if (it == prim_to_gpu.end()) continue;
                int mat_idx = gm.primitives[pi].material;
                if (mat_idx < 0) mat_idx = 0;
                scene.instances.push_back({it->second, (uint32_t)mat_idx, world, 0u});
            }
        }
        for (int child : node.children) traverse(child, world);
    };

    for (int root : gltf_scene.nodes) traverse(root, glm::mat4(1.0f));

    // ------------------------------------------------------------------ Sampler
    VkSamplerCreateInfo samp_ci{};
    samp_ci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samp_ci.magFilter    = VK_FILTER_LINEAR;
    samp_ci.minFilter    = VK_FILTER_LINEAR;
    samp_ci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samp_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samp_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samp_ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samp_ci.maxLod       = VK_LOD_CLAMP_NONE;
    if (vkCreateSampler(ctx.device.device, &samp_ci, nullptr, &scene.sampler) != VK_SUCCESS)
        throw std::runtime_error("GLTF sampler creation failed");

    return scene;
}
