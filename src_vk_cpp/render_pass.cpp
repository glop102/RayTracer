#include "render_pass.h"
#include "vk_context.h"
#include "swapchain.h"

#include <stdexcept>

RenderPass::RenderPass(VkContext& ctx, Swapchain& swapchain) {
    device = ctx.device.device;

    VkAttachmentDescription color_attachment{};
    color_attachment.format         = swapchain.image_format;
    color_attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_ref{};
    color_ref.attachment = 0;
    color_ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &color_ref;

    // Wait for the swapchain image to be readable before writing to it.
    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rp_info{};
    rp_info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.attachmentCount = 1;
    rp_info.pAttachments    = &color_attachment;
    rp_info.subpassCount    = 1;
    rp_info.pSubpasses      = &subpass;
    rp_info.dependencyCount = 1;
    rp_info.pDependencies   = &dep;

    if (vkCreateRenderPass(device, &rp_info, nullptr, &render_pass) != VK_SUCCESS)
        throw std::runtime_error("Render pass creation failed");

    framebuffers.resize(swapchain.image_views.size());
    for (size_t i = 0; i < swapchain.image_views.size(); i++) {
        VkImageView attachments[] = {swapchain.image_views[i]};

        VkFramebufferCreateInfo fb_info{};
        fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass      = render_pass;
        fb_info.attachmentCount = 1;
        fb_info.pAttachments    = attachments;
        fb_info.width           = swapchain.extent.width;
        fb_info.height          = swapchain.extent.height;
        fb_info.layers          = 1;

        if (vkCreateFramebuffer(device, &fb_info, nullptr, &framebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("Framebuffer creation failed");
    }
}

void RenderPass::rebuild_framebuffers(Swapchain& swapchain) {
    for (auto fb : framebuffers)
        vkDestroyFramebuffer(device, fb, nullptr);
    framebuffers.clear();

    framebuffers.resize(swapchain.image_views.size());
    for (size_t i = 0; i < swapchain.image_views.size(); i++) {
        VkImageView attachments[] = {swapchain.image_views[i]};

        VkFramebufferCreateInfo fb_info{};
        fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass      = render_pass;
        fb_info.attachmentCount = 1;
        fb_info.pAttachments    = attachments;
        fb_info.width           = swapchain.extent.width;
        fb_info.height          = swapchain.extent.height;
        fb_info.layers          = 1;

        if (vkCreateFramebuffer(device, &fb_info, nullptr, &framebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("Framebuffer creation failed");
    }
}

RenderPass::~RenderPass() {
    for (auto fb : framebuffers)
        vkDestroyFramebuffer(device, fb, nullptr);
    vkDestroyRenderPass(device, render_pass, nullptr);
}
