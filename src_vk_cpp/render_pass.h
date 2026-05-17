#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>
#include <vector>

struct VkContext;
struct Swapchain;

// Single-subpass render pass with color + depth attachments.
// The VkRenderPass object is created once; framebuffers and the depth image
// are recreated (via rebuild_framebuffers) whenever the swapchain resizes.
struct RenderPass {
    VkRenderPass               render_pass  = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;

    VkDevice      device    = VK_NULL_HANDLE;
    VmaAllocator  allocator = VK_NULL_HANDLE;

    VkImage       depth_image = VK_NULL_HANDLE;
    VmaAllocation depth_alloc = {};
    VkImageView   depth_view  = VK_NULL_HANDLE;

    RenderPass(VkContext& ctx, Swapchain& swapchain);
    ~RenderPass();

    void rebuild_framebuffers(Swapchain& swapchain);

    RenderPass(const RenderPass&)            = delete;
    RenderPass& operator=(const RenderPass&) = delete;
};
