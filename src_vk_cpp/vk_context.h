#pragma once
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

struct GLFWwindow;

// Owns the Vulkan instance, surface, physical/logical device, queues,
// command pool, VMA allocator, and RT extension function pointers.
struct VkContext {
    vkb::Instance       instance;
    VkSurfaceKHR        surface          = VK_NULL_HANDLE;
    vkb::PhysicalDevice physical_device;
    vkb::Device         device;
    VmaAllocator        allocator        = VK_NULL_HANDLE;

    VkQueue  graphics_queue        = VK_NULL_HANDLE;
    uint32_t graphics_queue_family = 0;
    VkQueue  present_queue         = VK_NULL_HANDLE;
    uint32_t present_queue_family  = 0;

    VkCommandPool command_pool = VK_NULL_HANDLE;

    // RT pipeline properties (handle sizes, alignment) queried at init.
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_pipeline_props{};

    // KHR extension function pointers — loaded after logical device creation.
    PFN_vkCreateAccelerationStructureKHR          pfn_vkCreateAccelerationStructureKHR          = nullptr;
    PFN_vkDestroyAccelerationStructureKHR         pfn_vkDestroyAccelerationStructureKHR         = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR   pfn_vkGetAccelerationStructureBuildSizesKHR   = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR       pfn_vkCmdBuildAccelerationStructuresKHR       = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR pfn_vkGetAccelerationStructureDeviceAddressKHR = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR            pfn_vkCreateRayTracingPipelinesKHR            = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR      pfn_vkGetRayTracingShaderGroupHandlesKHR      = nullptr;
    PFN_vkCmdTraceRaysKHR                         pfn_vkCmdTraceRaysKHR                         = nullptr;
    PFN_vkGetMemoryFdKHR                          pfn_vkGetMemoryFdKHR                          = nullptr;

    explicit VkContext(GLFWwindow* window);
    ~VkContext();

    VkContext(const VkContext&)            = delete;
    VkContext& operator=(const VkContext&) = delete;

    // One-shot command buffer helpers for uploads and AS builds.
    VkCommandBuffer begin_one_shot() const;
    void            end_one_shot(VkCommandBuffer cmd) const;
};
