#pragma once
#include <vulkan/vulkan.h>
#include <optional>
#include <vector>

struct VkContext;

// Owns all per-frame synchronisation: per-image command buffers, render-finished
// semaphores, in-flight fences, and the acquire semaphore free-list.
//
// Usage each frame:
//   auto [image_index, cmd] = frame_sync.acquire(swapchain.handle);
//   // record into cmd ...
//   frame_sync.submit_and_present(ctx.graphics_queue, ctx.present_queue, swapchain.handle);
struct FrameSync {
    struct Frame {
        uint32_t        image_index;
        VkCommandBuffer cmd;
    };

    // Returns nullopt if the swapchain is out of date and must be recreated.
    std::optional<Frame> acquire(VkSwapchainKHR swapchain);
    // Returns the VkResult from vkQueuePresentKHR (caller checks SUBOPTIMAL/OUT_OF_DATE).
    VkResult submit_and_present(VkQueue graphics, VkQueue present, VkSwapchainKHR swapchain);

    FrameSync(VkContext& ctx, uint32_t image_count);
    ~FrameSync();

    FrameSync(const FrameSync&)            = delete;
    FrameSync& operator=(const FrameSync&) = delete;

private:
    VkDevice device = VK_NULL_HANDLE;

    std::vector<VkCommandBuffer> cmd_bufs;
    std::vector<VkSemaphore>     render_finished;
    std::vector<VkFence>         in_flight;
    std::vector<VkSemaphore>     acquire_sems;      // full pool, owned for cleanup
    std::vector<VkSemaphore>     image_acquire_sem; // which sem is bound to each image slot
    std::vector<VkSemaphore>     free_acquire_sems; // available semaphores for next acquire

    uint32_t    current_image   = 0;
    VkSemaphore current_acquire = VK_NULL_HANDLE;
};
