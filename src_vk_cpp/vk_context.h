#pragma once
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>

struct GLFWwindow;

// Owns the Vulkan instance, surface, physical/logical device, queues,
// command pool, and VMA allocator. Requires RT extensions at device creation
// so GPU support is confirmed before any later stage work begins.
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

    explicit VkContext(GLFWwindow* window);
    ~VkContext();

    VkContext(const VkContext&)            = delete;
    VkContext& operator=(const VkContext&) = delete;
};
