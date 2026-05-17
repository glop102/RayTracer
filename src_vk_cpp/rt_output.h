#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

struct VkContext;

// Owns the storage image that the raygen shader writes to, plus the descriptor
// pool and set that binds the TLAS (binding 0) and image (binding 1).
// Must be recreated on swapchain resize (image dimensions must match).
struct RtOutput {
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool      descriptor_pool       = VK_NULL_HANDLE;
    VkDescriptorSet       descriptor_set        = VK_NULL_HANDLE;

    VkImage       image = VK_NULL_HANDLE;
    VmaAllocation alloc = {};
    VkImageView   view  = VK_NULL_HANDLE;

    RtOutput(VkContext& ctx, VkExtent2D extent, VkAccelerationStructureKHR tlas);
    ~RtOutput();

    // Recreate the storage image for a new extent and rebind all descriptors.
    void recreate(VkContext& ctx, VkExtent2D extent, VkAccelerationStructureKHR tlas);

    RtOutput(const RtOutput&)            = delete;
    RtOutput& operator=(const RtOutput&) = delete;

private:
    VkDevice     device    = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;

    void create_image(VkContext& ctx, VkExtent2D extent);
    void destroy_image();
    void write_descriptors(VkAccelerationStructureKHR tlas);
};
