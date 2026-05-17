// VMA_IMPLEMENTATION must appear in exactly one TU.
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <VkBootstrap.h>
#include <GLFW/glfw3.h>

#include "vk_context.h"

#include <stdexcept>

VkContext::VkContext(GLFWwindow* window) {
    // ------------------------------------------------------------------ Instance
    auto inst_ret = vkb::InstanceBuilder{}
        .set_app_name("vulkan-rt")
        .request_validation_layers()
        .use_default_debug_messenger()
        .require_api_version(1, 2, 0)
        .build();
    if (!inst_ret)
        throw std::runtime_error("Instance creation failed: " + inst_ret.error().message());
    instance = inst_ret.value();

    // ------------------------------------------------------------------ Surface
    if (glfwCreateWindowSurface(instance.instance, window, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("Window surface creation failed");

    // ------------------------------------------------------------------ RT feature structs
    // Chained now so device creation fails fast if the GPU does not support RT.
    // Feature bits are all VK_FALSE here; Stage 2 will query and enable them.
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_features{};
    rt_pipeline_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR as_features{};
    as_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;

    // ------------------------------------------------------------------ Physical device
    // Require RT extensions so an incapable GPU is caught here rather than later.
    // VK_KHR_DEFERRED_HOST_OPERATIONS is a prerequisite for acceleration structures.
    // buffer_device_address and descriptor_indexing are Vulkan 1.2 core, no extension needed.
    auto phys_ret = vkb::PhysicalDeviceSelector{instance}
        .set_surface(surface)
        .set_minimum_version(1, 2)
        .add_required_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
        .add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
        .add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
        .select();
    if (!phys_ret)
        throw std::runtime_error("Physical device selection failed: " + phys_ret.error().message());
    physical_device = phys_ret.value();

    // ------------------------------------------------------------------ Logical device
    auto dev_ret = vkb::DeviceBuilder{physical_device}
        .add_pNext(&rt_pipeline_features)
        .add_pNext(&as_features)
        .build();
    if (!dev_ret)
        throw std::runtime_error("Logical device creation failed: " + dev_ret.error().message());
    device = dev_ret.value();

    // ------------------------------------------------------------------ Queues
    auto gq = device.get_queue(vkb::QueueType::graphics);
    if (!gq) throw std::runtime_error("Failed to get graphics queue");
    graphics_queue        = gq.value();
    graphics_queue_family = device.get_queue_index(vkb::QueueType::graphics).value();

    auto pq = device.get_queue(vkb::QueueType::present);
    if (!pq) throw std::runtime_error("Failed to get present queue");
    present_queue        = pq.value();
    present_queue_family = device.get_queue_index(vkb::QueueType::present).value();

    // ------------------------------------------------------------------ Command pool
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = graphics_queue_family;
    if (vkCreateCommandPool(device.device, &pool_info, nullptr, &command_pool) != VK_SUCCESS)
        throw std::runtime_error("Command pool creation failed");

    // ------------------------------------------------------------------ VMA
    VmaVulkanFunctions vma_fns{};
    vma_fns.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vma_fns.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo alloc_info{};
    alloc_info.physicalDevice   = physical_device.physical_device;
    alloc_info.device           = device.device;
    alloc_info.instance         = instance.instance;
    alloc_info.vulkanApiVersion = VK_API_VERSION_1_2;
    alloc_info.pVulkanFunctions = &vma_fns;
    if (vmaCreateAllocator(&alloc_info, &allocator) != VK_SUCCESS)
        throw std::runtime_error("VMA allocator creation failed");
}

VkContext::~VkContext() {
    if (allocator)    vmaDestroyAllocator(allocator);
    if (command_pool) vkDestroyCommandPool(device.device, command_pool, nullptr);
    vkb::destroy_device(device);
    vkb::destroy_surface(instance, surface);
    vkb::destroy_instance(instance);
}
