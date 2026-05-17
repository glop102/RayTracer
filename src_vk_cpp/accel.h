#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <vector>

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
AccelStructure build_blas(VkContext& ctx, Mesh& mesh);

// One entry in the TLAS instance list.
struct TlasInstance {
    AccelStructure* blas;
    glm::mat4       transform;    // object-to-world (column-major GLM convention)
    uint32_t        custom_index; // gl_InstanceCustomIndexEXT (24-bit)
};

// Build a TLAS containing the given set of instances.
AccelStructure build_tlas(VkContext& ctx, const std::vector<TlasInstance>& instances);
