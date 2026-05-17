#pragma once
#include <VkBootstrap.h>
#include <vector>

struct VkContext;

struct Swapchain {
    vkb::Swapchain           swapchain;
    VkSwapchainKHR           handle       = VK_NULL_HANDLE;
    std::vector<VkImage>     images;
    std::vector<VkImageView> image_views;
    VkFormat                 image_format = VK_FORMAT_UNDEFINED;
    VkExtent2D               extent       = {0, 0};

    VkDevice device = VK_NULL_HANDLE;

    Swapchain(VkContext& ctx, uint32_t width, uint32_t height);
    ~Swapchain();

    void recreate(VkContext& ctx, uint32_t width, uint32_t height);

    Swapchain(const Swapchain&)            = delete;
    Swapchain& operator=(const Swapchain&) = delete;
};
