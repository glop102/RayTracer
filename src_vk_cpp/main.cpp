#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <stdexcept>

#include "vk_context.h"
#include "swapchain.h"
#include "camera.h"
#include "frame_sync.h"
#include "mesh.h"
#include "accel.h"
#include "scene_data.h"
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
        Mesh mesh {ctx, "bunny/reconstruction/bun_zipper_res2.ply"};
        Mesh box  {ctx, glm::vec3{-0.55f, -0.03f, -0.20f},
                        glm::vec3{ 0.55f,  0.22f,  0.20f}};
        // Area light: flat box above the scene, bottom face illuminates downward.
        Mesh light{ctx, glm::vec3{-0.4f, 0.33f, -0.25f},
                        glm::vec3{ 0.4f, 0.35f,  0.25f}};
        // Floor: large flat slab. Top face (y=-0.04) sits 1cm below the glass box
        // bottom (y=-0.03) to avoid coplanar surface ambiguity.
        Mesh floor{ctx, glm::vec3{-2.0f, -0.06f, -1.5f},
                        glm::vec3{ 2.0f, -0.04f,  1.5f}};

        AccelStructure blas       = build_blas(ctx, mesh);
        AccelStructure glass_blas = build_blas(ctx, box, false); // non-opaque for any-hit
        AccelStructure light_blas = build_blas(ctx, light);
        AccelStructure floor_blas = build_blas(ctx, floor);

        // Three bunnies side by side, each with a different material.
        // Glass box surrounds them (hit group 1). Area light sits above.
        auto translate = [](float tx, float ty, float tz) {
            glm::mat4 m(1.0f);
            m[3] = glm::vec4(tx, ty, tz, 1.0f);
            return m;
        };
        std::vector<TlasInstance> tlas_instances = {
            {&blas,       translate(-0.35f, 0.0f, 0.0f), 0, 0}, // left   — opaque
            {&blas,       glm::mat4(1.0f),               1, 0}, // centre — opaque
            {&blas,       translate( 0.35f, 0.0f, 0.0f), 2, 0}, // right  — opaque
            {&glass_blas, glm::mat4(1.0f),               3, 1}, // glass box — hit group 1
            {&light_blas, glm::mat4(1.0f),               4, 0}, // area light — opaque emissive
            {&floor_blas, glm::mat4(1.0f),               5, 0}, // floor — opaque diffuse
        };
        AccelStructure tlas = build_tlas(ctx, tlas_instances);

        std::vector<GpuMeshRef> mesh_refs = {
            {mesh.vertex_addr,  mesh.index_addr },  // mesh 0: bunny
            {box.vertex_addr,   box.index_addr  },  // mesh 1: glass box
            {light.vertex_addr, light.index_addr},  // mesh 2: area light
            {floor.vertex_addr, floor.index_addr},  // mesh 3: floor
        };
        std::vector<GpuMaterial> materials = {
            // diffuse              roughness  specular              ior   emissive          _pad1  absorption        _pad2
            {{0.75f,0.75f,0.75f}, 0.15f, {0.9f, 0.9f, 0.9f},  0.0f, {0.0f,0.0f,0.0f}, 0.0f, {0.0f,0.0f,0.0f}, 0.0f}, // 0: AluminiumDull
            {{0.80f,0.15f,0.10f}, 0.92f, {0.5f, 0.5f, 0.5f},  0.0f, {0.0f,0.0f,0.0f}, 0.0f, {0.0f,0.0f,0.0f}, 0.0f}, // 1: MatteRed
            {{0.80f,0.60f,0.20f}, 0.02f, {1.0f, 0.9f, 0.5f},  0.0f, {0.0f,0.0f,0.0f}, 0.0f, {0.0f,0.0f,0.0f}, 0.0f}, // 2: GoldMirror
            {{0.0f, 0.0f, 0.0f }, 0.0f,  {0.04f,0.04f,0.04f}, 1.5f, {0.0f,0.0f,0.0f}, 0.0f, {0.8f,0.2f,0.6f}, 0.0f}, // 3: Glass (teal tint)
            {{0.0f, 0.0f, 0.0f }, 1.0f,  {0.0f, 0.0f, 0.0f},  0.0f, {4.0f,3.5f,2.5f}, 0.0f, {0.0f,0.0f,0.0f}, 0.0f}, // 4: AreaLight
            {{0.6f, 0.6f, 0.58f}, 1.0f,  {0.0f, 0.0f, 0.0f},  0.0f, {0.0f,0.0f,0.0f}, 0.0f, {0.0f,0.0f,0.0f}, 0.0f}, // 5: Floor (pale concrete)
        };
        std::vector<GpuInstanceData> instance_data = {
            {0, 0, {0, 0}},  // instance 0 — left   bunny, AluminiumDull
            {0, 1, {0, 0}},  // instance 1 — centre bunny, MatteRed
            {0, 2, {0, 0}},  // instance 2 — right  bunny, GoldMirror
            {1, 3, {0, 0}},  // instance 3 — glass box,    Glass
            {2, 4, {0, 0}},  // instance 4 — area light,   AreaLight
            {3, 5, {0, 0}},  // instance 5 — floor,        Floor
        };

        // Build the NEE light list by scanning all instances for emissive materials.
        // meshes_by_index must match the mesh_index values in instance_data.
        std::vector<SceneInstance> scene_instances;
        scene_instances.reserve(tlas_instances.size());
        for (size_t i = 0; i < tlas_instances.size(); i++) {
            uint32_t iid = tlas_instances[i].custom_index;
            scene_instances.push_back({tlas_instances[i].transform,
                                       instance_data[iid].mesh_index,
                                       instance_data[iid].material_index});
        }
        const std::vector<const Mesh*> meshes_by_index = {&mesh, &box, &light, &floor};
        auto light_triangles = extract_light_triangles(scene_instances, meshes_by_index, materials);

        SceneData scene_data{ctx, mesh_refs, materials, instance_data, light_triangles};

        // SSBOs listed in binding order (2, 3, 4, 5 …).
        // Add new SSBOs here; RtOutput wires up bindings automatically.
        const GpuBuffer ssbos[] = {
            scene_data.mesh_refs,
            scene_data.materials,
            scene_data.instances,
            scene_data.light_triangles,
        };
        RtOutput rt_output{ctx, swapchain.extent, tlas.handle, ssbos};
        RtPipeline rt_pipeline{ctx, rt_output.descriptor_set_layout};

        uint32_t frame_index = 0;

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
                frame_index = 0;
                continue;
            }

            auto frame_opt = frame_sync.acquire(swapchain.handle);
            if (!frame_opt) { app.resize_needed = true; continue; }
            auto [image_index, cmd] = *frame_opt;

            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS)
                throw std::runtime_error("Failed to begin command buffer");

            // Bind RT pipeline and push camera parameters.
            if (camera.consume_moved()) frame_index = 0;

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline.pipeline);
            RtCameraPush push      = camera.rt_push(swapchain.extent);
            push.frame_index       = frame_index++;
            push.num_light_tris    = scene_data.light_count;
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
