#include "swapchain.h"
#include "vk_context.h"

#include <stdexcept>

Swapchain::Swapchain(VkContext& ctx, uint32_t width, uint32_t height) {
    device = ctx.device.device;

    auto swap_ret = vkb::SwapchainBuilder{ctx.device}
        .set_desired_format({VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        .set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
        .add_fallback_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .build();
    if (!swap_ret)
        throw std::runtime_error("Swapchain creation failed: " + swap_ret.error().message());
    swapchain = swap_ret.value();
    handle    = swapchain.swapchain;

    auto imgs = swapchain.get_images();
    if (!imgs) throw std::runtime_error("Failed to get swapchain images");
    images = imgs.value();

    auto views = swapchain.get_image_views();
    if (!views) throw std::runtime_error("Failed to get swapchain image views");
    image_views = views.value();

    image_format = swapchain.image_format;
    extent       = swapchain.extent;
}

Swapchain::~Swapchain() {
    for (auto view : image_views)
        vkDestroyImageView(device, view, nullptr);
    vkb::destroy_swapchain(swapchain);
}
