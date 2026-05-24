#pragma once
#include "denoiser.h"
#include <OpenImageDenoise/oidn.hpp>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

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

    VkDevice         vk_device_       = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VmaAllocator     allocator_       = VK_NULL_HANDLE;
    PFN_vkGetMemoryFdKHR pfn_get_fd_  = nullptr;
    uint32_t         w_ = 0, h_ = 0;

    VkImage color_image_  = VK_NULL_HANDLE;
    VkImage albedo_image_ = VK_NULL_HANDLE;
    VkImage normal_image_ = VK_NULL_HANDLE;

    VkImage       display_image_ = VK_NULL_HANDLE;
    VmaAllocation display_alloc_ = {};

    // Device-local buffers exported to OIDN/HIP via OpaqueFD — no CPU copies needed.
    struct ExportBuf {
        VkBuffer        buf     = VK_NULL_HANDLE;
        VkDeviceMemory  mem     = VK_NULL_HANDLE;
        oidn::BufferRef oidn_buf;
    };
    ExportBuf color_buf_, albedo_buf_, normal_buf_, output_buf_;

    ExportBuf make_export_buf(VkBufferUsageFlags usage) const;
    void      destroy_resources();
};
