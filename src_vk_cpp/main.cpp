#include <GLFW/glfw3.h>
#include <stdexcept>

#include "vk_context.h"
#include "swapchain.h"
#include "camera.h"
#include "frame_sync.h"
#include "mesh.h"
#include "accel.h"
#include "rt_output.h"
#include "rt_pipeline.h"

static constexpr uint32_t WIDTH  = 1280;
static constexpr uint32_t HEIGHT = 720;

// Transition a single image layout with a pipeline barrier.
static void image_barrier(VkCommandBuffer cmd, VkImage image,
                           VkAccessFlags src_access, VkAccessFlags dst_access,
                           VkImageLayout old_layout, VkImageLayout new_layout,
                           VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier b{};
    b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask       = src_access;
    b.dstAccessMask       = dst_access;
    b.oldLayout           = old_layout;
    b.newLayout           = new_layout;
    b.image               = image;
    b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

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

    {
        VkContext ctx{window};
        Swapchain swapchain{ctx, WIDTH, HEIGHT};
        Camera    camera{ctx};
        app.camera = &camera;
        FrameSync frame_sync{ctx, static_cast<uint32_t>(swapchain.images.size())};
        Mesh      mesh{ctx, "bunny/reconstruction/bun_zipper_res2.ply"};

        // Build acceleration structures — driver manages the BVH internally.
        AccelStructure blas = build_blas(ctx, mesh);
        AccelStructure tlas = build_tlas(ctx, blas);

        RtOutput   rt_output {ctx, swapchain.extent, tlas.handle};
        RtPipeline rt_pipeline{ctx, rt_output.descriptor_set_layout};

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            int fw, fh;
            glfwGetFramebufferSize(window, &fw, &fh);
            if (fw == 0 || fh == 0) continue;

            if (app.resize_needed) {
                app.resize_needed = false;
                vkDeviceWaitIdle(ctx.device.device);
                swapchain.recreate(ctx, static_cast<uint32_t>(fw), static_cast<uint32_t>(fh));
                rt_output.recreate(ctx, swapchain.extent, tlas.handle);
                continue;
            }

            auto frame_opt = frame_sync.acquire(swapchain.handle);
            if (!frame_opt) { app.resize_needed = true; continue; }
            auto [image_index, cmd] = *frame_opt;

            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS)
                throw std::runtime_error("Failed to begin command buffer");

            // Bind RT pipeline and push camera parameters.
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline.pipeline);
            RtCameraPush push = camera.rt_push(swapchain.extent);
            vkCmdPushConstants(cmd, rt_pipeline.layout,
                               VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(push), &push);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                    rt_pipeline.layout, 0, 1, &rt_output.descriptor_set, 0, nullptr);

            // Trace rays — raygen runs one invocation per pixel.
            ctx.pfn_vkCmdTraceRaysKHR(cmd,
                &rt_pipeline.raygen_region,
                &rt_pipeline.miss_region,
                &rt_pipeline.hit_region,
                &rt_pipeline.callable_region,
                swapchain.extent.width, swapchain.extent.height, 1);

            // Transition storage image GENERAL → TRANSFER_SRC for blit.
            image_barrier(cmd, rt_output.image,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT);

            // Transition swapchain image UNDEFINED → TRANSFER_DST.
            image_barrier(cmd, swapchain.images[image_index],
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

            // Blit storage image to swapchain image (same size → no filter needed).
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.srcOffsets[1]  = {(int32_t)swapchain.extent.width, (int32_t)swapchain.extent.height, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.dstOffsets[1]  = {(int32_t)swapchain.extent.width, (int32_t)swapchain.extent.height, 1};
            vkCmdBlitImage(cmd,
                rt_output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                swapchain.images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blit, VK_FILTER_NEAREST);

            // Transition swapchain image TRANSFER_DST → PRESENT_SRC.
            image_barrier(cmd, swapchain.images[image_index],
                VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

            // Transition storage image TRANSFER_SRC → GENERAL for next frame.
            image_barrier(cmd, rt_output.image,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);

            if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
                throw std::runtime_error("Failed to end command buffer");

            VkResult present_result = frame_sync.submit_and_present(
                ctx.graphics_queue, ctx.present_queue, swapchain.handle);
            if (present_result == VK_ERROR_OUT_OF_DATE_KHR ||
                present_result == VK_SUBOPTIMAL_KHR)
                app.resize_needed = true;
        }

        vkDeviceWaitIdle(ctx.device.device);

    } // all Vulkan objects destroyed in reverse construction order before glfwDestroyWindow

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
