#include <GLFW/glfw3.h>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "vk_context.h"
#include "swapchain.h"
#include "render_pass.h"
#include "pipeline.h"

static constexpr uint32_t WIDTH  = 1280;
static constexpr uint32_t HEIGHT = 720;

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan RT", nullptr, nullptr);

    // Scope ensures all Vulkan destructors run before glfwDestroyWindow.
    // GLFW requires the surface be destroyed first.
    {
        VkContext  ctx{window};
        Swapchain  swapchain{ctx, WIDTH, HEIGHT};
        RenderPass render_pass{ctx, swapchain};
        Pipeline   pipeline{ctx, render_pass, swapchain.extent};

        const uint32_t image_count = static_cast<uint32_t>(swapchain.images.size());
        VkDevice       dev         = ctx.device.device;

        // All per-image resources are indexed by image_index, not by a frame slot.
        //
        // The presentation engine considers a semaphore "in use" for image N until N
        // is re-acquired via vkAcquireNextImageKHR. This applies to BOTH the acquire
        // semaphore (signaled by vkAcquireNextImageKHR, waited by submit) AND the
        // render_finished semaphore (signaled by submit, waited by vkQueuePresentKHR).
        // Indexing either by frame-in-flight instead of image_index causes
        // VUID-vkQueueSubmit-pSignalSemaphores-00067 whenever frames_in_flight < image_count.
        //
        // By indexing by image_index, every semaphore is only reused for image N at the
        // point N is re-acquired — which is guaranteed to be after N's prior presentation
        // completed, ending the engine's association with that semaphore.

        std::vector<VkCommandBuffer> cmd_bufs(image_count);
        std::vector<VkSemaphore>     render_finished(image_count);
        std::vector<VkFence>         in_flight(image_count);

        {
            VkCommandBufferAllocateInfo alloc{};
            alloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            alloc.commandPool        = ctx.command_pool;
            alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            alloc.commandBufferCount = image_count;
            if (vkAllocateCommandBuffers(dev, &alloc, cmd_bufs.data()) != VK_SUCCESS)
                throw std::runtime_error("Command buffer allocation failed");
        }

        VkSemaphoreCreateInfo sem_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkFenceCreateInfo     fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (uint32_t i = 0; i < image_count; i++) {
            vkCreateSemaphore(dev, &sem_info,   nullptr, &render_finished[i]);
            vkCreateFence    (dev, &fence_info, nullptr, &in_flight[i]);
        }

        // Acquire semaphore free-list: image_count + 1 semaphores so there is always
        // a free one regardless of how many images are simultaneously in flight.
        // After re-acquiring image N, the semaphore previously bound to N is returned
        // to the free list — the re-acquire ends the engine's prior association.
        std::vector<VkSemaphore> acquire_sems(image_count + 1);
        for (auto& s : acquire_sems)
            vkCreateSemaphore(dev, &sem_info, nullptr, &s);

        std::vector<VkSemaphore> image_acquire_sem(image_count, VK_NULL_HANDLE);
        std::vector<VkSemaphore> free_acquire_sems(acquire_sems.begin(), acquire_sems.end());

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            // Pick a free acquire semaphore before knowing which image we'll get.
            VkSemaphore acquire_sem = free_acquire_sems.back();
            free_acquire_sems.pop_back();

            uint32_t image_index = 0;
            vkAcquireNextImageKHR(dev, swapchain.swapchain.swapchain,
                                  UINT64_MAX, acquire_sem, VK_NULL_HANDLE, &image_index);

            // The old semaphore for this slot is now free (image was just re-acquired).
            if (image_acquire_sem[image_index] != VK_NULL_HANDLE)
                free_acquire_sems.push_back(image_acquire_sem[image_index]);
            image_acquire_sem[image_index] = acquire_sem;

            // Wait for the previous frame that rendered to this image slot to finish.
            vkWaitForFences(dev, 1, &in_flight[image_index], VK_TRUE, UINT64_MAX);
            vkResetFences  (dev, 1, &in_flight[image_index]);

            VkCommandBuffer cmd = cmd_bufs[image_index];
            vkResetCommandBuffer(cmd, 0);

            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            vkBeginCommandBuffer(cmd, &begin);

            VkClearValue clear_color{{{0.01f, 0.01f, 0.02f, 1.0f}}};
            VkRenderPassBeginInfo rp_begin{};
            rp_begin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rp_begin.renderPass        = render_pass.render_pass;
            rp_begin.framebuffer       = render_pass.framebuffers[image_index];
            rp_begin.renderArea.offset = {0, 0};
            rp_begin.renderArea.extent = swapchain.extent;
            rp_begin.clearValueCount   = 1;
            rp_begin.pClearValues      = &clear_color;

            vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRenderPass(cmd);
            vkEndCommandBuffer(cmd);

            VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkSubmitInfo submit{};
            submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.waitSemaphoreCount   = 1;
            submit.pWaitSemaphores      = &acquire_sem;
            submit.pWaitDstStageMask    = &wait_stage;
            submit.commandBufferCount   = 1;
            submit.pCommandBuffers      = &cmd;
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores    = &render_finished[image_index];
            vkQueueSubmit(ctx.graphics_queue, 1, &submit, in_flight[image_index]);

            VkSwapchainKHR   sc = swapchain.swapchain.swapchain;
            VkPresentInfoKHR present_info{};
            present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            present_info.waitSemaphoreCount = 1;
            present_info.pWaitSemaphores    = &render_finished[image_index];
            present_info.swapchainCount     = 1;
            present_info.pSwapchains        = &sc;
            present_info.pImageIndices      = &image_index;
            vkQueuePresentKHR(ctx.present_queue, &present_info);
        }

        vkDeviceWaitIdle(dev);

        for (uint32_t i = 0; i < image_count; i++) {
            vkDestroySemaphore(dev, render_finished[i], nullptr);
            vkDestroyFence    (dev, in_flight[i],       nullptr);
        }
        for (auto s : acquire_sems)
            vkDestroySemaphore(dev, s, nullptr);

    } // render_pass, swapchain, ctx destroyed — surface before window

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
