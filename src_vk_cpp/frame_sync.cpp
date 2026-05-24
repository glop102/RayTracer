#include "frame_sync.h"
#include "vk_context.h"

#include <optional>
#include <stdexcept>

FrameSync::FrameSync(VkContext& ctx, uint32_t image_count) {
    device = ctx.device.device;

    cmd_bufs.resize(image_count);
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool        = ctx.command_pool;
    alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = image_count;
    if (vkAllocateCommandBuffers(device, &alloc, cmd_bufs.data()) != VK_SUCCESS)
        throw std::runtime_error("Command buffer allocation failed");

    VkSemaphoreCreateInfo sem_info{};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    render_finished.resize(image_count);
    in_flight.resize(image_count);
    image_acquire_sem.resize(image_count, VK_NULL_HANDLE);

    for (uint32_t i = 0; i < image_count; i++) {
        if (vkCreateSemaphore(device, &sem_info,   nullptr, &render_finished[i]) != VK_SUCCESS ||
            vkCreateFence    (device, &fence_info, nullptr, &in_flight[i])        != VK_SUCCESS)
            throw std::runtime_error("Sync object creation failed");
    }

    acquire_sems.resize(image_count + 1);
    for (auto& s : acquire_sems) {
        if (vkCreateSemaphore(device, &sem_info, nullptr, &s) != VK_SUCCESS)
            throw std::runtime_error("Acquire semaphore creation failed");
    }
    free_acquire_sems.assign(acquire_sems.begin(), acquire_sems.end());
}

FrameSync::~FrameSync() {
    for (auto s : render_finished) vkDestroySemaphore(device, s, nullptr);
    for (auto f : in_flight)       vkDestroyFence    (device, f, nullptr);
    for (auto s : acquire_sems)    vkDestroySemaphore(device, s, nullptr);
    // cmd_bufs are implicitly freed when the command pool is destroyed in VkContext
}

std::optional<FrameSync::Frame> FrameSync::acquire(VkSwapchainKHR swapchain) {
    current_acquire = free_acquire_sems.back();
    free_acquire_sems.pop_back();

    VkResult acquire_result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                                     current_acquire, VK_NULL_HANDLE, &current_image);
    if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
        free_acquire_sems.push_back(current_acquire);
        return std::nullopt;
    }
    if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("Failed to acquire swapchain image");

    if (image_acquire_sem[current_image] != VK_NULL_HANDLE)
        free_acquire_sems.push_back(image_acquire_sem[current_image]);
    image_acquire_sem[current_image] = current_acquire;

    if (vkWaitForFences(device, 1, &in_flight[current_image], VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        throw std::runtime_error("Fence wait failed");
    if (vkResetFences(device, 1, &in_flight[current_image]) != VK_SUCCESS)
        throw std::runtime_error("Fence reset failed");

    VkCommandBuffer cmd = cmd_bufs[current_image];
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS)
        throw std::runtime_error("Command buffer reset failed");

    return Frame{current_image, cmd};
}

VkResult FrameSync::submit_and_present(VkQueue graphics, VkQueue present, VkSwapchainKHR swapchain) {
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo submit{};
    submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount   = 1;
    submit.pWaitSemaphores      = &current_acquire;
    submit.pWaitDstStageMask    = &wait_stage;
    submit.commandBufferCount   = 1;
    submit.pCommandBuffers      = &cmd_bufs[current_image];
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores    = &render_finished[current_image];
    if (vkQueueSubmit(graphics, 1, &submit, in_flight[current_image]) != VK_SUCCESS)
        throw std::runtime_error("Queue submit failed");

    VkPresentInfoKHR present_info{};
    present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores    = &render_finished[current_image];
    present_info.swapchainCount     = 1;
    present_info.pSwapchains        = &swapchain;
    present_info.pImageIndices      = &current_image;
    return vkQueuePresentKHR(present, &present_info);
}
