#pragma once
#include <vulkan/vulkan.h>
#include <vector>

struct VkContext;
struct Swapchain;

// Single-subpass render pass with one color attachment (clear on load,
// store on exit). Framebuffers are one-per-swapchain-image.
// Depth attachment is added in milestone 3.
struct RenderPass {
    VkRenderPass               render_pass  = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;

    VkDevice device = VK_NULL_HANDLE;

    RenderPass(VkContext& ctx, Swapchain& swapchain);
    ~RenderPass();

    void rebuild_framebuffers(Swapchain& swapchain);

    RenderPass(const RenderPass&)            = delete;
    RenderPass& operator=(const RenderPass&) = delete;
};
