#include <GLFW/glfw3.h>
#include <stdexcept>

#include "vk_context.h"
#include "swapchain.h"
#include "render_pass.h"
#include "pipeline.h"
#include "frame_sync.h"

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
        Pipeline   pipeline{ctx, render_pass};
        FrameSync  frame_sync{ctx, static_cast<uint32_t>(swapchain.images.size())};

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            auto [image_index, cmd] = frame_sync.acquire(swapchain.handle);

            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS)
                throw std::runtime_error("Failed to begin command buffer");

            VkClearValue clear_color{{{0.01f, 0.01f, 0.02f, 1.0f}}};
            VkRenderPassBeginInfo rp_begin{};
            rp_begin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rp_begin.renderPass        = render_pass.render_pass;
            rp_begin.framebuffer       = render_pass.framebuffers[image_index];
            rp_begin.renderArea.extent = swapchain.extent;
            rp_begin.clearValueCount   = 1;
            rp_begin.pClearValues      = &clear_color;

            vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);

            VkViewport viewport{};
            viewport.width    = static_cast<float>(swapchain.extent.width);
            viewport.height   = static_cast<float>(swapchain.extent.height);
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(cmd, 0, 1, &viewport);

            VkRect2D scissor{scissor.offset = {}, scissor.extent = swapchain.extent};
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            vkCmdDraw(cmd, 3, 1, 0, 0);
            vkCmdEndRenderPass(cmd);
            if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
                throw std::runtime_error("Failed to end command buffer");

            frame_sync.submit_and_present(ctx.graphics_queue, ctx.present_queue, swapchain.handle);
        }

        vkDeviceWaitIdle(ctx.device.device);

    } // frame_sync, pipeline, render_pass, swapchain, ctx destroyed in reverse order

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
