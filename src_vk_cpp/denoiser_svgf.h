#pragma once
#include "denoiser.h"
#include <vk_mem_alloc.h>

struct VkContext;

// GPU À-trous wavelet denoiser (4 passes, 5×5 dilated kernel).
// All work happens in record_pre(); execute() and record_post() are no-ops.
// output_image() is in TRANSFER_SRC_OPTIMAL after record_pre() returns.
class SvgfDenoiser final : public IDenoiser {
public:
    SvgfDenoiser() = default;
    ~SvgfDenoiser() override;

    void    setup(VkContext& ctx, uint32_t w, uint32_t h,
                  VkImage color, VkImage albedo, VkImage normal) override;
    VkImage output_image() const override;
    void    record_pre(VkCommandBuffer cmd) override;

private:
    static constexpr int NUM_PASSES = 4;

    VkDevice     device_    = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    uint32_t w_ = 0, h_ = 0;

    VkImage color_image_  = VK_NULL_HANDLE;
    VkImage albedo_image_ = VK_NULL_HANDLE;
    VkImage normal_image_ = VK_NULL_HANDLE;

    // Views over the G-buffer images owned by RtOutput.
    VkImageView color_view_  = VK_NULL_HANDLE;
    VkImageView albedo_view_ = VK_NULL_HANDLE;
    VkImageView normal_view_ = VK_NULL_HANDLE;

    // Ping-pong buffers written alternately by each pass.
    struct FilterBuf {
        VkImage       image = VK_NULL_HANDLE;
        VmaAllocation alloc = {};
        VkImageView   view  = VK_NULL_HANDLE;
    };
    FilterBuf bufs_[2] = {};

    VkDescriptorSetLayout dsl_              = VK_NULL_HANDLE;
    VkDescriptorPool      pool_             = VK_NULL_HANDLE;
    VkDescriptorSet       dsets_[NUM_PASSES] = {};

    VkPipelineLayout pipe_layout_ = VK_NULL_HANDLE;
    VkPipeline       pipeline_    = VK_NULL_HANDLE;

    void create_pipeline(VkContext& ctx); // once per lifetime
    void destroy_images();                // called on every resize
    FilterBuf   make_filter_buf() const;
    VkImageView make_view(VkImage image) const;
};
