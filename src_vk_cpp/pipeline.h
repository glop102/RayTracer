#pragma once
#include <vulkan/vulkan.h>

struct VkContext;
struct RenderPass;

struct Pipeline {
    VkPipelineLayout layout   = VK_NULL_HANDLE;
    VkPipeline       pipeline = VK_NULL_HANDLE;
    VkDevice         device   = VK_NULL_HANDLE;

    Pipeline(VkContext& ctx, RenderPass& render_pass, VkDescriptorSetLayout camera_layout);
    ~Pipeline();

    Pipeline(const Pipeline&)            = delete;
    Pipeline& operator=(const Pipeline&) = delete;
};
