#pragma once
#include "gpu_buffer.h"
#include "camera.h"
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <string>

struct VkContext;

// Manages the high-quality offline screenshot pipeline:
//   - f64 accumulation SSBO (no precision loss from running mean)
//   - HQ raygen shader (no Russian roulette)
//   - Resolve compute pass (sum ÷ N → f32 image for denoiser)
//   - PNG readback + save
struct ScreenshotMode {
    bool     active       = false;
    uint32_t samples_done = 0;
    uint32_t target       = 256;

    uint32_t width = 0, height = 0;

    // f64 per-pixel accumulation buffer (dvec4, row-major)
    GpuBuffer accum_buf;

    // Descriptor set (set=1) binding the accum buffer for the HQ raygen
    VkDescriptorSetLayout accum_dsl  = VK_NULL_HANDLE;
    VkDescriptorPool      accum_pool = VK_NULL_HANDLE;
    VkDescriptorSet       accum_set  = VK_NULL_HANDLE;

    // HQ ray tracing pipeline (raygen_hq.rgen + shared miss/hit shaders)
    VkPipelineLayout hq_layout   = VK_NULL_HANDLE;
    VkPipeline       hq_pipeline = VK_NULL_HANDLE;
    VkBuffer         hq_sbt_buf  = VK_NULL_HANDLE;
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

    // Host-visible staging buffer for PNG readback
    GpuBuffer readback_buf;

    // Call once after the RT pipeline and output are set up.
    // color_view: rt_output.view (used as resolve target).
    // rt_output_layout: rt_output.descriptor_set_layout (set 0 for the HQ pipeline).
    void setup(VkContext& ctx, VkExtent2D extent,
               VkDescriptorSetLayout rt_output_layout,
               VkImage color_image, VkImageView color_view);

    // Call again on swapchain/rt_output resize to resize buffers and rebind.
    void update_color_image(VkContext& ctx, VkExtent2D new_extent,
                            VkImage color_image, VkImageView color_view);

    // Start a new screenshot capture: zeroes accum buffer, resets counter.
    void begin(VkContext& ctx, uint32_t n_samples);

    // Record one sample dispatch into cmd (use instead of the regular raygen dispatch).
    void record_sample(VkCommandBuffer cmd, VkDescriptorSet rt_output_set,
                       const RtCameraPush& push, uint32_t w, uint32_t h);

    // After all samples: dispatch resolve compute, transitions out_image to GENERAL.
    void resolve(VkContext& ctx);

    // Readback from the given image (denoiser output or color image) and save PNG.
    // layout must be VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL or GENERAL.
    void save_png(VkContext& ctx, VkImage src_image, VkImageLayout src_layout,
                  const std::string& path);

    void destroy(VkContext& ctx);

private:
    VkDevice     device_      = VK_NULL_HANDLE;
    VmaAllocator allocator_   = VK_NULL_HANDLE;
    VkImage      color_image_ = VK_NULL_HANDLE;
    PFN_vkCmdTraceRaysKHR pfn_trace_ = nullptr;
};
