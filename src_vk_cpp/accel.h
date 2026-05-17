#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

struct VkContext;
struct Mesh;

// Owns a single VkAccelerationStructureKHR and its backing buffer.
// Move-only (destructor calls vkDestroyAccelerationStructureKHR).
struct AccelStructure {
    VkAccelerationStructureKHR handle  = VK_NULL_HANDLE;
    VkBuffer                   buf     = VK_NULL_HANDLE;
    VmaAllocation              alloc   = {};
    VkDeviceAddress            address = 0;  // for TLAS instance references

    AccelStructure() = default;
    ~AccelStructure();

    AccelStructure(AccelStructure&& o) noexcept;
    AccelStructure& operator=(AccelStructure&& o) noexcept;

    AccelStructure(const AccelStructure&)            = delete;
    AccelStructure& operator=(const AccelStructure&) = delete;

    // Internal fields — set by build_blas/build_tlas, used by destructor.
    VkDevice     device     = VK_NULL_HANDLE;
    VmaAllocator allocator  = VK_NULL_HANDLE;
    PFN_vkDestroyAccelerationStructureKHR pfn_destroy = nullptr;
};

// Build a BLAS for a single triangle mesh.
// The mesh buffers must have ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY and
// SHADER_DEVICE_ADDRESS usage flags (set by Mesh constructor).
AccelStructure build_blas(VkContext& ctx, Mesh& mesh);

// Build a TLAS with a single identity-transform instance pointing at blas.
AccelStructure build_tlas(VkContext& ctx, AccelStructure& blas);
