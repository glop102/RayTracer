#include "render_pass.h"
#include "vk_context.h"
#include "swapchain.h"

#include <stdexcept>

static constexpr VkFormat DEPTH_FORMAT = VK_FORMAT_D32_SFLOAT;

static void create_depth_resources(VkDevice device, VmaAllocator allocator, VkExtent2D extent,
                                   VkImage& image, VmaAllocation& alloc, VkImageView& view) {
    VkImageCreateInfo img_ci{};
    img_ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType     = VK_IMAGE_TYPE_2D;
    img_ci.format        = DEPTH_FORMAT;
    img_ci.extent        = {extent.width, extent.height, 1};
    img_ci.mipLevels     = 1;
    img_ci.arrayLayers   = 1;
    img_ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    img_ci.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_ci.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    if (vmaCreateImage(allocator, &img_ci, &alloc_ci, &image, &alloc, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Depth image creation failed");

    VkImageViewCreateInfo view_ci{};
    view_ci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_ci.image                           = image;
    view_ci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format                          = DEPTH_FORMAT;
    view_ci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    view_ci.subresourceRange.baseMipLevel   = 0;
    view_ci.subresourceRange.levelCount     = 1;
    view_ci.subresourceRange.baseArrayLayer = 0;
    view_ci.subresourceRange.layerCount     = 1;

    if (vkCreateImageView(device, &view_ci, nullptr, &view) != VK_SUCCESS)
        throw std::runtime_error("Depth image view creation failed");
}

RenderPass::RenderPass(VkContext& ctx, Swapchain& swapchain) {
    device    = ctx.device.device;
    allocator = ctx.allocator;

    // ------------------------------------------------------------------ Render pass
    VkAttachmentDescription attachments[2]{};

    // attachment 0: color
    attachments[0].format         = swapchain.image_format;
    attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // attachment 1: depth
    attachments[1].format         = DEPTH_FORMAT;
    attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depth_ref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &color_ref;
    subpass.pDepthStencilAttachment = &depth_ref;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rp_info{};
    rp_info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp_info.attachmentCount = 2;
    rp_info.pAttachments    = attachments;
    rp_info.subpassCount    = 1;
    rp_info.pSubpasses      = &subpass;
    rp_info.dependencyCount = 1;
    rp_info.pDependencies   = &dep;

    if (vkCreateRenderPass(device, &rp_info, nullptr, &render_pass) != VK_SUCCESS)
        throw std::runtime_error("Render pass creation failed");

    // ------------------------------------------------------------------ Depth image + framebuffers
    create_depth_resources(device, allocator, swapchain.extent,
                           depth_image, depth_alloc, depth_view);

    framebuffers.resize(swapchain.image_views.size());
    for (size_t i = 0; i < swapchain.image_views.size(); i++) {
        VkImageView atts[] = {swapchain.image_views[i], depth_view};

        VkFramebufferCreateInfo fb_info{};
        fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass      = render_pass;
        fb_info.attachmentCount = 2;
        fb_info.pAttachments    = atts;
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

    vkDestroyImageView(device, depth_view,  nullptr);
    vmaDestroyImage(allocator, depth_image, depth_alloc);
    depth_view  = VK_NULL_HANDLE;
    depth_image = VK_NULL_HANDLE;

    create_depth_resources(device, allocator, swapchain.extent,
                           depth_image, depth_alloc, depth_view);

    framebuffers.resize(swapchain.image_views.size());
    for (size_t i = 0; i < swapchain.image_views.size(); i++) {
        VkImageView atts[] = {swapchain.image_views[i], depth_view};

        VkFramebufferCreateInfo fb_info{};
        fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb_info.renderPass      = render_pass;
        fb_info.attachmentCount = 2;
        fb_info.pAttachments    = atts;
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
    vkDestroyImageView(device, depth_view,  nullptr);
    vmaDestroyImage(allocator, depth_image, depth_alloc);
    vkDestroyRenderPass(device, render_pass, nullptr);
}
