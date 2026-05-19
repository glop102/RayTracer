#include "scene_data.h"
#include "vk_context.h"

#include <cstring>
#include <stdexcept>

static void upload_ssbo(VkContext& ctx,
                        const void* data, VkDeviceSize size,
                        VkBuffer& out_buf, VmaAllocation& out_alloc) {
    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size  = size;
    ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
               VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info;
    if (vmaCreateBuffer(ctx.allocator, &ci, &ai, &out_buf, &out_alloc, &info) != VK_SUCCESS)
        throw std::runtime_error("Scene SSBO creation failed");
    std::memcpy(info.pMappedData, data, size);
}

SceneData::SceneData(VkContext& ctx,
                     const std::vector<GpuMeshRef>&       mesh_refs,
                     const std::vector<GpuMaterial>&      materials,
                     const std::vector<GpuInstanceData>&  instances,
                     const std::vector<GpuLightTriangle>& light_triangles) {
    device    = ctx.device.device;
    allocator = ctx.allocator;

    mesh_refs_range       = mesh_refs.size()       * sizeof(GpuMeshRef);
    materials_range       = materials.size()        * sizeof(GpuMaterial);
    instances_range       = instances.size()        * sizeof(GpuInstanceData);
    light_triangles_range = light_triangles.size()  * sizeof(GpuLightTriangle);
    light_count           = static_cast<uint32_t>(light_triangles.size());

    upload_ssbo(ctx, mesh_refs.data(),       mesh_refs_range,       mesh_refs_buf,       mesh_refs_alloc);
    upload_ssbo(ctx, materials.data(),       materials_range,       materials_buf,       materials_alloc);
    upload_ssbo(ctx, instances.data(),       instances_range,       instances_buf,       instances_alloc);
    upload_ssbo(ctx, light_triangles.data(), light_triangles_range, light_triangles_buf, light_triangles_alloc);
}

SceneData::~SceneData() {
    vmaDestroyBuffer(allocator, mesh_refs_buf,       mesh_refs_alloc);
    vmaDestroyBuffer(allocator, materials_buf,       materials_alloc);
    vmaDestroyBuffer(allocator, instances_buf,       instances_alloc);
    vmaDestroyBuffer(allocator, light_triangles_buf, light_triangles_alloc);
}
