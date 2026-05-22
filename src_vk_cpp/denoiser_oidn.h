#pragma once
#include "denoiser.h"
#include <OpenImageDenoise/oidn.hpp>
#include <vk_mem_alloc.h>

class OidnDenoiser final : public IDenoiser {
public:
    OidnDenoiser();
    ~OidnDenoiser() override;

    void    setup(VkContext& ctx, uint32_t w, uint32_t h,
                  VkImage color, VkImage albedo, VkImage normal) override;
    VkImage output_image() const override { return display_image_; }
    void    record_pre(VkCommandBuffer cmd) override;
    void    execute() override;
    void    record_post(VkCommandBuffer cmd) override;

private:
    oidn::DeviceRef device_;
    oidn::FilterRef filter_;

    VkDevice     vk_device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    uint32_t     w_ = 0, h_ = 0;

    VkImage color_image_  = VK_NULL_HANDLE;
    VkImage albedo_image_ = VK_NULL_HANDLE;
    VkImage normal_image_ = VK_NULL_HANDLE;

    VkImage       display_image_ = VK_NULL_HANDLE;
    VmaAllocation display_alloc_ = {};

    // OIDN-managed host buffers (Storage::Host → pinned memory, accessible by HIP and CPU)
    oidn::BufferRef color_oidn_buf_, albedo_oidn_buf_, normal_oidn_buf_, output_oidn_buf_;

    struct StagingBuf {
        VkBuffer      buf   = VK_NULL_HANDLE;
        VmaAllocation alloc = {};
        void*         ptr   = nullptr;
    };
    StagingBuf color_stg_, albedo_stg_, normal_stg_, output_stg_;

    StagingBuf make_staging(VkBufferUsageFlags usage) const;
    void       destroy_resources();
};
