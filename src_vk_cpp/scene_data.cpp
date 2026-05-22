#include "scene_data.h"
#include "gpu_buffer.h"
#include "vk_context.h"

std::vector<GpuLightTriangle> extract_light_triangles(
    const std::vector<SceneInstance>& instances,
    const std::vector<const Mesh*>&   meshes,
    const std::vector<GpuMaterial>&   materials) {

    std::vector<GpuLightTriangle> result;
    for (const auto& inst : instances) {
        const GpuMaterial& mat = materials[inst.material_index];
        if (mat.emissive.x == 0.0f && mat.emissive.y == 0.0f && mat.emissive.z == 0.0f)
            continue;

        const Mesh* mesh = meshes[inst.mesh_index];
        for (size_t t = 0; t < mesh->cpu_indices.size(); t += 3) {
            uint32_t i0 = mesh->cpu_indices[t];
            uint32_t i1 = mesh->cpu_indices[t + 1];
            uint32_t i2 = mesh->cpu_indices[t + 2];

            glm::vec3 v0 = glm::vec3(inst.transform * glm::vec4(mesh->cpu_positions[i0], 1.0f));
            glm::vec3 v1 = glm::vec3(inst.transform * glm::vec4(mesh->cpu_positions[i1], 1.0f));
            glm::vec3 v2 = glm::vec3(inst.transform * glm::vec4(mesh->cpu_positions[i2], 1.0f));

            result.push_back({v0, 0.0f, v1, 0.0f, v2, 0.0f, mat.emissive, 0.0f});
        }
    }

    // Build power-weighted CDF.  power_i = area_i * ||emission_i||.
    // cdf field holds the running normalized cumulative sum in [0,1].
    // select_weight = total_power / ||emission_i|| folds the PDF into the estimator
    // so the shader needs no separate total_power uniform.
    float total_power = 0.0f;
    for (auto& lt : result) {
        float area  = 0.5f * glm::length(glm::cross(lt.v1 - lt.v0, lt.v2 - lt.v0));
        float power = area * glm::length(lt.emission);
        lt.cdf       = power;   // temporarily store raw power; normalized below
        total_power += power;
    }
    if (total_power > 0.0f) {
        float running = 0.0f;
        for (auto& lt : result) {
            running      += lt.cdf;
            lt.cdf        = running / total_power;
            float em_len  = glm::length(lt.emission);
            lt.select_weight = (em_len > 0.0f) ? total_power / em_len : 0.0f;
        }
    }

    return result;
}

SceneData::SceneData(VkContext& ctx,
                     const std::vector<GpuMeshRef>&       mr,
                     const std::vector<GpuMaterial>&      mat,
                     const std::vector<GpuInstanceData>&  inst,
                     const std::vector<GpuLightTriangle>& lt) {
    allocator   = ctx.allocator;
    light_count = static_cast<uint32_t>(lt.size());

    mesh_refs       = upload_ssbo(ctx, mr.data(),   mr.size()   * sizeof(GpuMeshRef));
    materials       = upload_ssbo(ctx, mat.data(),  mat.size()  * sizeof(GpuMaterial));
    instances       = upload_ssbo(ctx, inst.data(), inst.size() * sizeof(GpuInstanceData));
    light_triangles = upload_ssbo(ctx, lt.data(),   lt.size()   * sizeof(GpuLightTriangle));
}

SceneData::~SceneData() {
    destroy_buffer(allocator, mesh_refs);
    destroy_buffer(allocator, materials);
    destroy_buffer(allocator, instances);
    destroy_buffer(allocator, light_triangles);
}
