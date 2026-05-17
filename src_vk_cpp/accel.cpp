#include "accel.h"
#include "vk_context.h"
#include "mesh.h"

#include <glm/gtc/type_ptr.hpp>

#include <cstring>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// AccelStructure lifecycle

AccelStructure::~AccelStructure() {
    if (handle != VK_NULL_HANDLE && pfn_destroy)
        pfn_destroy(device, handle, nullptr);
    if (buf != VK_NULL_HANDLE)
        vmaDestroyBuffer(allocator, buf, alloc);
}

AccelStructure::AccelStructure(AccelStructure&& o) noexcept
    : handle(o.handle), buf(o.buf), alloc(o.alloc), address(o.address),
      device(o.device), allocator(o.allocator), pfn_destroy(o.pfn_destroy) {
    o.handle      = VK_NULL_HANDLE;
    o.buf         = VK_NULL_HANDLE;
    o.device      = VK_NULL_HANDLE;
    o.allocator   = VK_NULL_HANDLE;
    o.pfn_destroy = nullptr;
}

AccelStructure& AccelStructure::operator=(AccelStructure&& o) noexcept {
    if (this != &o) {
        this->~AccelStructure();
        new (this) AccelStructure(std::move(o));
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Internal helpers

// Create a device-local buffer with SHADER_DEVICE_ADDRESS.
static VkBuffer make_device_buffer(VkContext& ctx,
                                   VkDeviceSize size,
                                   VkBufferUsageFlags usage,
                                   VmaAllocation& out_alloc) {
    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size  = size;
    ci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VkBuffer buf;
    if (vmaCreateBuffer(ctx.allocator, &ci, &ai, &buf, &out_alloc, nullptr) != VK_SUCCESS)
        throw std::runtime_error("AS buffer creation failed");
    return buf;
}

static VkDeviceAddress buffer_address(VkDevice dev, VkBuffer buf) {
    VkBufferDeviceAddressInfo info{};
    info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buf;
    return vkGetBufferDeviceAddress(dev, &info);
}

// Create an AccelerationStructureKHR backed by an already-allocated buffer.
static void create_as(VkContext& ctx,
                      AccelStructure& result,
                      VkAccelerationStructureTypeKHR type,
                      VkDeviceSize as_size) {
    result.device      = ctx.device.device;
    result.allocator   = ctx.allocator;
    result.pfn_destroy = ctx.pfn_vkDestroyAccelerationStructureKHR;

    result.buf = make_device_buffer(ctx, as_size,
                                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                    result.alloc);

    VkAccelerationStructureCreateInfoKHR as_ci{};
    as_ci.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    as_ci.buffer = result.buf;
    as_ci.size   = as_size;
    as_ci.type   = type;
    if (ctx.pfn_vkCreateAccelerationStructureKHR(ctx.device.device, &as_ci, nullptr, &result.handle) != VK_SUCCESS)
        throw std::runtime_error("vkCreateAccelerationStructureKHR failed");

    VkAccelerationStructureDeviceAddressInfoKHR addr_info{};
    addr_info.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addr_info.accelerationStructure = result.handle;
    result.address = ctx.pfn_vkGetAccelerationStructureDeviceAddressKHR(ctx.device.device, &addr_info);
}

// ---------------------------------------------------------------------------
// BLAS build

AccelStructure build_blas(VkContext& ctx, Mesh& mesh) {
    // Describe the triangle geometry from the mesh device buffers.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat  = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = mesh.vertex_addr;
    triangles.vertexStride  = sizeof(float) * 6;  // interleaved {pos, normal}
    triangles.maxVertex     = mesh.vertex_count - 1;
    triangles.indexType     = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress  = mesh.index_addr;

    VkAccelerationStructureGeometryKHR geom{};
    geom.sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.flags        = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.triangles = triangles;

    uint32_t prim_count = mesh.index_count / 3;

    VkAccelerationStructureBuildGeometryInfoKHR build_info{};
    build_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_info.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build_info.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = 1;
    build_info.pGeometries   = &geom;

    // Query required buffer sizes before allocating anything.
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    ctx.pfn_vkGetAccelerationStructureBuildSizesKHR(
        ctx.device.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info, &prim_count, &sizes);

    // Allocate the AS backing buffer and create the handle.
    AccelStructure result;
    create_as(ctx, result, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
              sizes.accelerationStructureSize);

    // Scratch buffer (transient — freed after build).
    VmaAllocation scratch_alloc{};
    VkBuffer scratch_buf = make_device_buffer(ctx, sizes.buildScratchSize,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                              scratch_alloc);

    build_info.dstAccelerationStructure  = result.handle;
    build_info.scratchData.deviceAddress = buffer_address(ctx.device.device, scratch_buf);

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = prim_count;
    const VkAccelerationStructureBuildRangeInfoKHR* range_ptr = &range;

    VkCommandBuffer cmd = ctx.begin_one_shot();
    ctx.pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_ptr);
    ctx.end_one_shot(cmd);

    vmaDestroyBuffer(ctx.allocator, scratch_buf, scratch_alloc);
    return result;
}

// ---------------------------------------------------------------------------
// TLAS build

AccelStructure build_tlas(VkContext& ctx, const std::vector<TlasInstance>& instances) {
    // Build the flat VkAccelerationStructureInstanceKHR array.
    // GLM is column-major; VkTransformMatrixKHR is row-major 3x4.
    // Transposing the GLM mat4 and copying the first 3 rows (12 floats) produces
    // the correct row-major layout expected by Vulkan.
    std::vector<VkAccelerationStructureInstanceKHR> vk_insts(instances.size());
    for (size_t i = 0; i < instances.size(); i++) {
        const TlasInstance& src = instances[i];
        VkAccelerationStructureInstanceKHR& dst = vk_insts[i];
        glm::mat4 T = glm::transpose(src.transform);
        std::memcpy(dst.transform.matrix, glm::value_ptr(T), sizeof(dst.transform.matrix));
        dst.instanceCustomIndex                    = src.custom_index & 0xFFFFFFu;
        dst.mask                                   = 0xFF;
        dst.instanceShaderBindingTableRecordOffset = 0;
        dst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        dst.accelerationStructureReference         = src.blas->address;
    }

    VkDeviceSize inst_size = vk_insts.size() * sizeof(VkAccelerationStructureInstanceKHR);

    // Upload via staging buffer — TLAS build reads from device-local memory.
    VkBufferCreateInfo stg_ci{};
    stg_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stg_ci.size  = inst_size;
    stg_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stg_ai{};
    stg_ai.usage = VMA_MEMORY_USAGE_AUTO;
    stg_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VkBuffer stg_buf; VmaAllocation stg_alloc;
    if (vmaCreateBuffer(ctx.allocator, &stg_ci, &stg_ai, &stg_buf, &stg_alloc, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Instance staging buffer creation failed");
    void* stg_mapped;
    vmaMapMemory(ctx.allocator, stg_alloc, &stg_mapped);
    std::memcpy(stg_mapped, vk_insts.data(), inst_size);
    vmaUnmapMemory(ctx.allocator, stg_alloc);

    VmaAllocation inst_alloc{};
    VkBuffer inst_buf = make_device_buffer(
        ctx, inst_size,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        inst_alloc);

    VkCommandBuffer cmd = ctx.begin_one_shot();
    VkBufferCopy region{0, 0, inst_size};
    vkCmdCopyBuffer(cmd, stg_buf, inst_buf, 1, &region);

    VkMemoryBarrier barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    VkAccelerationStructureGeometryInstancesDataKHR inst_data{};
    inst_data.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    inst_data.data.deviceAddress = buffer_address(ctx.device.device, inst_buf);

    VkAccelerationStructureGeometryKHR geom{};
    geom.sType               = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType        = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.flags               = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.instances  = inst_data;

    uint32_t inst_count = static_cast<uint32_t>(instances.size());

    VkAccelerationStructureBuildGeometryInfoKHR build_info{};
    build_info.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build_info.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build_info.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = 1;
    build_info.pGeometries   = &geom;

    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    ctx.pfn_vkGetAccelerationStructureBuildSizesKHR(
        ctx.device.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build_info, &inst_count, &sizes);

    AccelStructure result;
    create_as(ctx, result, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
              sizes.accelerationStructureSize);

    VmaAllocation scratch_alloc{};
    VkBuffer scratch_buf = make_device_buffer(ctx, sizes.buildScratchSize,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                              scratch_alloc);

    build_info.dstAccelerationStructure  = result.handle;
    build_info.scratchData.deviceAddress = buffer_address(ctx.device.device, scratch_buf);

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = inst_count;
    const VkAccelerationStructureBuildRangeInfoKHR* range_ptr = &range;

    ctx.pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &build_info, &range_ptr);
    ctx.end_one_shot(cmd);

    vmaDestroyBuffer(ctx.allocator, scratch_buf, scratch_alloc);
    vmaDestroyBuffer(ctx.allocator, inst_buf,    inst_alloc);
    vmaDestroyBuffer(ctx.allocator, stg_buf,     stg_alloc);
    return result;
}
