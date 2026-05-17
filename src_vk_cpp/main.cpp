#include <GLFW/glfw3.h>
#include <stdexcept>

#include "vk_context.h"
#include "swapchain.h"
#include "render_pass.h"
#include "pipeline.h"
#include "frame_sync.h"
#include "mesh.h"
#include "camera.h"

static constexpr uint32_t WIDTH  = 1280;
static constexpr uint32_t HEIGHT = 720;

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan RT", nullptr, nullptr);

    struct AppState { bool resize_needed = false; Camera* camera = nullptr; };
    AppState app;
    glfwSetWindowUserPointer(window, &app);

    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* w, int, int) {
        static_cast<AppState*>(glfwGetWindowUserPointer(w))->resize_needed = true;
    });
    glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int btn, int action, int) {
        auto* s = static_cast<AppState*>(glfwGetWindowUserPointer(w));
        if (s->camera) s->camera->on_mouse_button(btn, action);
    });
    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double x, double y) {
        auto* s = static_cast<AppState*>(glfwGetWindowUserPointer(w));
        if (s->camera) s->camera->on_cursor_pos(x, y);
    });
    glfwSetScrollCallback(window, [](GLFWwindow* w, double, double dy) {
        auto* s = static_cast<AppState*>(glfwGetWindowUserPointer(w));
        if (s->camera) s->camera->on_scroll(dy);
    });

    // Scope ensures all Vulkan destructors run before glfwDestroyWindow.
    // GLFW requires the surface be destroyed first.
    {
        VkContext  ctx{window};
        Swapchain  swapchain{ctx, WIDTH, HEIGHT};
        RenderPass render_pass{ctx, swapchain};
        Camera     camera{ctx};
        app.camera = &camera;
        Pipeline   pipeline{ctx, render_pass, camera.descriptor_set_layout};
        FrameSync  frame_sync{ctx, static_cast<uint32_t>(swapchain.images.size())};
        Mesh       mesh{ctx, "bunny/reconstruction/bun_zipper_res2.ply"};

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            int fw, fh;
            glfwGetFramebufferSize(window, &fw, &fh);
            if (fw == 0 || fh == 0) continue; // minimized — nothing to render

            if (app.resize_needed) {
                app.resize_needed = false;
                vkDeviceWaitIdle(ctx.device.device);
                swapchain.recreate(ctx, static_cast<uint32_t>(fw), static_cast<uint32_t>(fh));
                render_pass.rebuild_framebuffers(swapchain);
                continue;
            }

            camera.update(swapchain.extent);

            auto frame_opt = frame_sync.acquire(swapchain.handle);
            if (!frame_opt) { app.resize_needed = true; continue; }
            auto [image_index, cmd] = *frame_opt;

            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS)
                throw std::runtime_error("Failed to begin command buffer");

            VkClearValue clears[2]{};
            clears[0].color        = {{0.01f, 0.01f, 0.02f, 1.0f}};
            clears[1].depthStencil = {1.0f, 0};

            VkRenderPassBeginInfo rp_begin{};
            rp_begin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rp_begin.renderPass        = render_pass.render_pass;
            rp_begin.framebuffer       = render_pass.framebuffers[image_index];
            rp_begin.renderArea.extent = swapchain.extent;
            rp_begin.clearValueCount   = 2;
            rp_begin.pClearValues      = clears;

            vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);

            VkViewport viewport{};
            viewport.width    = static_cast<float>(swapchain.extent.width);
            viewport.height   = static_cast<float>(swapchain.extent.height);
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(cmd, 0, 1, &viewport);

            VkRect2D scissor{scissor.offset = {}, scissor.extent = swapchain.extent};
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline.layout, 0, 1, &camera.descriptor_set, 0, nullptr);

            VkDeviceSize zero = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertex_buf, &zero);
            vkCmdBindIndexBuffer(cmd, mesh.index_buf, 0, VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexed(cmd, mesh.index_count, 1, 0, 0, 0);
            vkCmdEndRenderPass(cmd);
            if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
                throw std::runtime_error("Failed to end command buffer");

            VkResult present_result = frame_sync.submit_and_present(
                ctx.graphics_queue, ctx.present_queue, swapchain.handle);
            if (present_result == VK_ERROR_OUT_OF_DATE_KHR ||
                present_result == VK_SUBOPTIMAL_KHR)
                app.resize_needed = true;
        }

        vkDeviceWaitIdle(ctx.device.device);

    } // frame_sync, pipeline, render_pass, swapchain, ctx destroyed in reverse order

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
