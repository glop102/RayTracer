#include "scene_data.h"
#include "gpu_buffer.h"
#include "vk_context.h"

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
