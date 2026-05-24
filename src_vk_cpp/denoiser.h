#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>

struct VkContext;

enum class DenoiserMode { None, OIDN, SVGF };

class IDenoiser {
public:
    virtual ~IDenoiser();

    // (Re)initialise for the given resolution and G-buffer images (all in GENERAL layout).
    virtual void setup(VkContext& ctx, uint32_t w, uint32_t h,
                       VkImage color, VkImage albedo, VkImage normal) = 0;

    // Image to blit to the swapchain. In TRANSFER_SRC_OPTIMAL after record_post returns.
    virtual VkImage output_image() const = 0;

    // Record GPU readback (CPU denoisers) or full denoising dispatch (GPU denoisers) into cmd.
    virtual void record_pre(VkCommandBuffer cmd) = 0;

    // CPU-side denoising work. No-op for GPU denoisers.
    virtual void execute() {}

    // Record staging→output upload and transition to TRANSFER_SRC_OPTIMAL. No-op for GPU denoisers.
    virtual void record_post(VkCommandBuffer /*cmd*/) {}
};
