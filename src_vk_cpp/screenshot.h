#pragma once
#include "gpu_buffer.h"
#include "camera.h"
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <string>

struct VkContext;

// Manages the high-quality offline screenshot pipeline:
//   - f64 accumulation SSBO (no precision loss from running mean)
//   - HQ raygen shader (no Russian roulette); writes albedo+normal G-buffers
//     into ss_albedo_image / ss_normal_image (set=1 bindings 1/2)
//   - Resolve compute pass (sum ÷ N → f32 image)
//   - Optional OIDN denoising via a temporary OidnDenoiser in main.cpp
//   - PNG readback + save
struct ScreenshotMode {
    bool     active       = false;
    bool     custom_res   = false;  // true when screenshot res != swapchain res
    uint32_t samples_done = 0;
    uint32_t target       = 1024;

    // Swapchain resolution (updated on resize via update_swapchain_size)
    uint32_t width = 0, height = 0;

    // Screenshot resolution for the current/next capture
    uint32_t ss_width = 0, ss_height = 0;

    // f64 per-pixel accumulation buffer (dvec4, row-major, sized to ss_width×ss_height)
    GpuBuffer accum_buf;

    // Descriptor set (set=1): binding 0 = accum SSBO, 1 = albedo image, 2 = normal image
    VkDescriptorSetLayout accum_dsl  = VK_NULL_HANDLE;
    VkDescriptorPool      accum_pool = VK_NULL_HANDLE;
    VkDescriptorSet       accum_set  = VK_NULL_HANDLE;

    // HQ ray tracing pipeline (raygen_hq.rgen + shared miss/hit shaders)
    VkPipelineLayout hq_layout    = VK_NULL_HANDLE;
    VkPipeline       hq_pipeline  = VK_NULL_HANDLE;
    VkBuffer         hq_sbt_buf   = VK_NULL_HANDLE;
    VmaAllocation    hq_sbt_alloc = {};
    VkStridedDeviceAddressRegionKHR hq_raygen_region   = {};
    VkStridedDeviceAddressRegionKHR hq_miss_region     = {};
    VkStridedDeviceAddressRegionKHR hq_hit_region      = {};
    VkStridedDeviceAddressRegionKHR hq_callable_region = {};

    // Resolve compute pipeline (dvec4 buffer → f32 storage image)
    VkDescriptorSetLayout resolve_dsl    = VK_NULL_HANDLE;
    VkDescriptorPool      resolve_pool   = VK_NULL_HANDLE;
    VkDescriptorSet       resolve_set    = VK_NULL_HANDLE;
    VkPipelineLayout      resolve_layout = VK_NULL_HANDLE;
    VkPipeline            resolve_pl     = VK_NULL_HANDLE;

    // Color output image for custom-res captures (same-res resolves into rt_output.image)
    VkImage       ss_image = VK_NULL_HANDLE;
    VmaAllocation ss_alloc = {};
    VkImageView   ss_view  = VK_NULL_HANDLE;

    // G-buffer images for OIDN denoising — always at ss_width×ss_height
    VkImage       ss_albedo_image = VK_NULL_HANDLE;
    VmaAllocation ss_albedo_alloc = {};
    VkImageView   ss_albedo_view  = VK_NULL_HANDLE;
    VkImage       ss_normal_image = VK_NULL_HANDLE;
    VmaAllocation ss_normal_alloc = {};
    VkImageView   ss_normal_view  = VK_NULL_HANDLE;

    // Host-visible staging buffer for PNG readback (sized to ss_width×ss_height)
    GpuBuffer readback_buf;

    // Call once after the RT pipeline and output are set up.
    void setup(VkContext& ctx, VkExtent2D extent,
               VkDescriptorSetLayout rt_output_layout,
               VkImage color_image, VkImageView color_view);

    // Call on swapchain/rt_output resize. Aborts any in-progress capture and
    // updates the stored swapchain size and same-res resolve target.
    void update_swapchain_size(VkContext& ctx, VkExtent2D new_extent,
                               VkImage color_image, VkImageView color_view);

    // Start a new screenshot capture at the given resolution (0 = use swapchain size).
    void begin(VkContext& ctx, uint32_t n_samples, uint32_t ss_w = 0, uint32_t ss_h = 0);

    // Record one sample dispatch into cmd.
    void record_sample(VkCommandBuffer cmd, VkDescriptorSet rt_output_set,
                       const RtCameraPush& push);

    // After all samples: dispatch resolve compute.
    // Same-res: resolves into rt_output.image. Custom-res: resolves into ss_image.
    void resolve(VkContext& ctx);

    // Readback from the given image and save PNG.
    // layout must be VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL or GENERAL.
    void save_png(VkContext& ctx, VkImage src_image, VkImageLayout src_layout,
                  const std::string& path);

    void destroy(VkContext& ctx);

private:
    VkDevice     device_         = VK_NULL_HANDLE;
    VmaAllocator allocator_      = VK_NULL_HANDLE;
    VkImage      rt_color_image_ = VK_NULL_HANDLE;
    VkImageView  rt_color_view_  = VK_NULL_HANDLE;
    PFN_vkCmdTraceRaysKHR pfn_trace_ = nullptr;

    void alloc_capture_buffers(VkContext& ctx);
    void update_resolve_target(VkImageView view);
    void destroy_ss_image();
    void destroy_gbuf_images();
};
