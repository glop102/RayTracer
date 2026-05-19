// VMA_IMPLEMENTATION must appear in exactly one TU.
#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <VkBootstrap.h>
#include <GLFW/glfw3.h>

#include "vk_context.h"

#include <stdexcept>

#define LOAD_PFN(name) \
    pfn_##name = reinterpret_cast<PFN_##name>( \
        vkGetDeviceProcAddr(device.device, #name)); \
    if (!pfn_##name) throw std::runtime_error("Failed to load " #name)

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

    // ------------------------------------------------------------------ RT feature structs (with bits enabled)
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_features{};
    rt_pipeline_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rt_pipeline_features.rayTracingPipeline = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR as_features{};
    as_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    as_features.accelerationStructure = VK_TRUE;

    // bufferDeviceAddress and descriptorIndexing are Vulkan 1.2 core.
    VkPhysicalDeviceVulkan12Features vk12_features{};
    vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk12_features.bufferDeviceAddress                       = VK_TRUE;
    vk12_features.descriptorIndexing                        = VK_TRUE;
    vk12_features.runtimeDescriptorArray                    = VK_TRUE;
    vk12_features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    vk12_features.descriptorBindingVariableDescriptorCount  = VK_TRUE;
    vk12_features.descriptorBindingPartiallyBound           = VK_TRUE;

    // shaderInt64 is a 1.0 core feature; route it through VkPhysicalDeviceFeatures2
    // in the pNext chain (incompatible with pEnabledFeatures, which we leave NULL).
    VkPhysicalDeviceFeatures2 base_features{};
    base_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    base_features.features.shaderInt64 = VK_TRUE;

    // ------------------------------------------------------------------ Physical device
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
        .add_pNext(&vk12_features)
        .add_pNext(&base_features)
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

    // ------------------------------------------------------------------ VMA (with buffer device address)
    VmaVulkanFunctions vma_fns{};
    vma_fns.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vma_fns.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo alloc_info{};
    alloc_info.physicalDevice   = physical_device.physical_device;
    alloc_info.device           = device.device;
    alloc_info.instance         = instance.instance;
    alloc_info.vulkanApiVersion = VK_API_VERSION_1_2;
    alloc_info.pVulkanFunctions = &vma_fns;
    alloc_info.flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    if (vmaCreateAllocator(&alloc_info, &allocator) != VK_SUCCESS)
        throw std::runtime_error("VMA allocator creation failed");

    // ------------------------------------------------------------------ RT extension function pointers
    LOAD_PFN(vkCreateAccelerationStructureKHR);
    LOAD_PFN(vkDestroyAccelerationStructureKHR);
    LOAD_PFN(vkGetAccelerationStructureBuildSizesKHR);
    LOAD_PFN(vkCmdBuildAccelerationStructuresKHR);
    LOAD_PFN(vkGetAccelerationStructureDeviceAddressKHR);
    LOAD_PFN(vkCreateRayTracingPipelinesKHR);
    LOAD_PFN(vkGetRayTracingShaderGroupHandlesKHR);
    LOAD_PFN(vkCmdTraceRaysKHR);

    // ------------------------------------------------------------------ RT pipeline properties
    rt_pipeline_props.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &rt_pipeline_props;
    vkGetPhysicalDeviceProperties2(physical_device.physical_device, &props2);
}

VkContext::~VkContext() {
    if (allocator)    vmaDestroyAllocator(allocator);
    if (command_pool) vkDestroyCommandPool(device.device, command_pool, nullptr);
    vkb::destroy_device(device);
    vkb::destroy_surface(instance, surface);
    vkb::destroy_instance(instance);
}

VkCommandBuffer VkContext::begin_one_shot() const {
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool        = command_pool;
    alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;

    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(device.device, &alloc, &cmd) != VK_SUCCESS)
        throw std::runtime_error("One-shot command buffer allocation failed");

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    return cmd;
}

void VkContext::end_one_shot(VkCommandBuffer cmd) const {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd;
    vkQueueSubmit(graphics_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphics_queue);

    vkFreeCommandBuffers(device.device, command_pool, 1, &cmd);
}
