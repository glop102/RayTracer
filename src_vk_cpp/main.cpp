#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <optional>
#include <stdexcept>
#include <vector>

#include "vk_context.h"
#include "swapchain.h"
#include "camera.h"
#include "frame_sync.h"
#include "mesh.h"
#include "accel.h"
#include "scene_data.h"
#include "gltf_scene.h"
#include "rt_output.h"
#include "rt_pipeline.h"
#include "denoiser.h"

static constexpr uint32_t WIDTH  = 1280;
static constexpr uint32_t HEIGHT = 720;

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

int main(int argc, char* argv[]) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan RT", nullptr, nullptr);

    struct AppState { bool resize_needed = false; Camera* camera = nullptr; Denoiser* denoiser = nullptr; };
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
    glfwSetKeyCallback(window, [](GLFWwindow* w, int key, int, int action, int mods) {
        auto* s = static_cast<AppState*>(glfwGetWindowUserPointer(w));
        if (key == GLFW_KEY_D && action == GLFW_PRESS && (mods & GLFW_MOD_ALT) && s->denoiser)
            s->denoiser->enabled = !s->denoiser->enabled;
    });
    {
        VkContext ctx{window};
        Swapchain swapchain{ctx, WIDTH, HEIGHT};
        Camera    camera;
        camera.set_window(window);
        app.camera = &camera;
        FrameSync frame_sync{ctx, static_cast<uint32_t>(swapchain.images.size())};

        // ================================================================
        // Scene setup — GLTF or hard-coded
        // ================================================================

        // Lifetime owners for hard-coded-path mesh objects
        std::vector<std::unique_ptr<Mesh>> hc_meshes;

        // Optional GLTF scene (owns textures + sampler; meshes stay via hc_meshes or gltf)
        std::optional<LoadedScene> gltf;

        // Built below regardless of path
        std::vector<AccelStructure>  blas_list;
        std::vector<TlasInstance>    tlas_insts;
        std::vector<GpuMeshRef>      mesh_refs_data;
        std::vector<GpuMaterial>     materials_data;
        std::vector<GpuInstanceData> inst_data;
        std::vector<SceneInstance>   scene_insts;
        std::vector<const Mesh*>     meshes_by_idx;
        std::vector<VkImageView>     tex_views;
        VkSampler                    tex_sampler = VK_NULL_HANDLE;

        if (argc > 1) {
            // ---------------------------------------------------------- GLTF path
            // Pull camera back so the Cornell box is fully visible from outside.
            camera.reset_pose({0.0f, 0.5f, 4.5f}, std::numbers::pi_v<float>, -0.1f);

            gltf.emplace(load_gltf(ctx, argv[1]));

            // Reserve for GLTF meshes + area light + 5 Cornell walls so later
            // push_back calls don't reallocate and invalidate TlasInstance pointers.
            blas_list.reserve(gltf->meshes.size() + 6);

            for (auto& mp : gltf->meshes)
                blas_list.push_back(build_blas(ctx, *mp));

            for (size_t i = 0; i < gltf->instances.size(); i++) {
                auto& gi = gltf->instances[i];
                tlas_insts.push_back({&blas_list[gi.mesh_index], gi.transform,
                                      (uint32_t)i, gi.sbt_record_offset});
            }
            for (auto& mp : gltf->meshes) {
                mesh_refs_data.push_back({mp->vertex_addr, mp->index_addr});
                meshes_by_idx.push_back(mp.get());
            }
            materials_data = gltf->materials;
            for (auto& gi : gltf->instances) {
                inst_data.push_back({gi.mesh_index, gi.material_index, {0,0}});
                scene_insts.push_back({gi.transform, gi.mesh_index, gi.material_index});
            }
            for (auto& t : gltf->textures) tex_views.push_back(t.view);
            tex_sampler = gltf->sampler;

            // Helper: add a static box mesh with the given material.
            // Appends to all relevant data structures in lock-step.
            auto add_box = [&](glm::vec3 mn, glm::vec3 mx, GpuMaterial mat) {
                hc_meshes.push_back(std::make_unique<Mesh>(ctx, mn, mx));
                auto* m = hc_meshes.back().get();
                uint32_t mesh_idx = (uint32_t)mesh_refs_data.size();
                uint32_t mat_idx  = (uint32_t)materials_data.size();
                uint32_t inst_idx = (uint32_t)inst_data.size();
                blas_list.push_back(build_blas(ctx, *m));
                mesh_refs_data.push_back({m->vertex_addr, m->index_addr});
                meshes_by_idx.push_back(m);
                materials_data.push_back(mat);
                tlas_insts.push_back({&blas_list.back(), glm::mat4(1.0f), inst_idx, 0});
                inst_data.push_back({mesh_idx, mat_idx, {0, 0}});
                scene_insts.push_back({glm::mat4(1.0f), mesh_idx, mat_idx});
            };

            // Shared material helper: diffuse-only, no textures.
            auto diffuse_mat = [](glm::vec3 col) {
                GpuMaterial m{};
                m.diffuse    = col;
                m.roughness  = 1.0f;
                m.specular   = {0.04f, 0.04f, 0.04f};
                m.diffuse_tex = m.mr_tex = m.normal_tex = m.emissive_tex = -1;
                return m;
            };

            // Cornell box — 5 m cube (X ±2.5, Y −1→4, Z ±2.5), open front (+Z face).
            // Thin slabs so the inward-facing normal is the one rays hit.
            add_box({-2.5f, -1.05f, -2.5f}, { 2.5f, -1.0f,  2.5f}, diffuse_mat({0.90f, 0.90f, 0.90f})); // floor
            add_box({-2.5f,  4.0f,  -2.5f}, { 2.5f,  4.05f, 2.5f}, diffuse_mat({0.90f, 0.90f, 0.90f})); // ceiling
            add_box({-2.5f, -1.0f,  -2.55f},{ 2.5f,  4.05f,-2.5f}, diffuse_mat({0.90f, 0.90f, 0.90f})); // back wall
            add_box({-2.55f,-1.0f,  -2.5f}, {-2.5f,  4.05f, 2.5f}, diffuse_mat({0.75f, 0.10f, 0.10f})); // left wall  (red)
            add_box({ 2.5f, -1.0f,  -2.5f}, { 2.55f, 4.05f, 2.5f}, diffuse_mat({0.10f, 0.65f, 0.10f})); // right wall (green)

            // Area light: warm-white panel recessed into the ceiling.
            GpuMaterial lm{};
            lm.emissive    = {8.0f, 7.5f, 6.5f};
            lm.roughness   = 1.0f;
            lm.diffuse_tex = lm.mr_tex = lm.normal_tex = lm.emissive_tex = -1;
            add_box({-0.8f, 3.85f, -0.8f}, {0.8f, 3.95f, 0.8f}, lm);

        } else {
            // ---------------------------------------------------------- Hard-coded path
            hc_meshes.push_back(std::make_unique<Mesh>(ctx, "bunny/reconstruction/bun_zipper_res2.ply"));
            hc_meshes.push_back(std::make_unique<Mesh>(ctx,
                glm::vec3{-0.55f, -0.03f, -0.20f}, glm::vec3{ 0.55f,  0.22f,  0.20f}));
            hc_meshes.push_back(std::make_unique<Mesh>(ctx,
                glm::vec3{-0.4f,  0.33f, -0.25f},  glm::vec3{ 0.4f,   0.35f,  0.25f}));
            hc_meshes.push_back(std::make_unique<Mesh>(ctx,
                glm::vec3{-2.0f, -0.06f, -1.5f},   glm::vec3{ 2.0f,  -0.04f,  1.5f}));

            auto* bunny = hc_meshes[0].get();
            auto* box   = hc_meshes[1].get();
            auto* light = hc_meshes[2].get();
            auto* floor = hc_meshes[3].get();

            blas_list.push_back(build_blas(ctx, *bunny));
            blas_list.push_back(build_blas(ctx, *box, false));  // non-opaque for glass any-hit
            blas_list.push_back(build_blas(ctx, *light));
            blas_list.push_back(build_blas(ctx, *floor));

            auto translate = [](float tx, float ty, float tz) {
                glm::mat4 m(1.0f);
                m[3] = glm::vec4(tx, ty, tz, 1.0f);
                return m;
            };
            tlas_insts = {
                {&blas_list[0], translate(-0.35f,0,0), 0, 0},
                {&blas_list[0], glm::mat4(1),          1, 0},
                {&blas_list[0], translate( 0.35f,0,0), 2, 0},
                {&blas_list[1], glm::mat4(1),          3, 1},
                {&blas_list[2], glm::mat4(1),          4, 0},
                {&blas_list[3], glm::mat4(1),          5, 0},
            };

            mesh_refs_data = {
                {bunny->vertex_addr, bunny->index_addr},
                {box->vertex_addr,   box->index_addr},
                {light->vertex_addr, light->index_addr},
                {floor->vertex_addr, floor->index_addr},
            };
            meshes_by_idx = {bunny, box, light, floor};

            using M = GpuMaterial;
            materials_data = {
                // diffuse              rough  specular              ior  emissive  metallic  absorption  _pad2  tex indices
                M{{.75f,.75f,.75f}, .15f, {.9f,.9f,.9f},  0, {0,0,0}, 0, {0,0,0}, 0, -1,-1,-1,-1}, // AluminiumDull
                M{{.80f,.15f,.10f}, .92f, {.5f,.5f,.5f},  0, {0,0,0}, 0, {0,0,0}, 0, -1,-1,-1,-1}, // MatteRed
                M{{.80f,.60f,.20f}, .02f, {1.f,.9f,.5f},  0, {0,0,0}, 0, {0,0,0}, 0, -1,-1,-1,-1}, // GoldMirror
                M{{0,0,0},          0,   {.04f,.04f,.04f},1.5f,{0,0,0},0,{.8f,.2f,.6f},0,-1,-1,-1,-1},// Glass
                M{{0,0,0},          1,   {0,0,0},         0, {4,3.5f,2.5f},0,{0,0,0},0,-1,-1,-1,-1},  // AreaLight
                M{{.6f,.6f,.58f},   1,   {0,0,0},         0, {0,0,0}, 0, {0,0,0}, 0, -1,-1,-1,-1},    // Floor
            };
            inst_data = {
                {0,0,{0,0}}, {0,1,{0,0}}, {0,2,{0,0}},  // 3 bunnies
                {1,3,{0,0}},                               // glass box
                {2,4,{0,0}},                               // area light
                {3,5,{0,0}},                               // floor
            };
            scene_insts = {
                {translate(-0.35f,0,0), 0, 0},
                {glm::mat4(1),          0, 1},
                {translate( 0.35f,0,0), 0, 2},
                {glm::mat4(1),          1, 3},
                {glm::mat4(1),          2, 4},
                {glm::mat4(1),          3, 5},
            };
        }

        // ================================================================
        // Common: TLAS, scene data, RT output, pipeline
        // ================================================================

        AccelStructure tlas = build_tlas(ctx, tlas_insts);

        auto light_triangles = extract_light_triangles(scene_insts, meshes_by_idx, materials_data);
        SceneData scene_data{ctx, mesh_refs_data, materials_data, inst_data, light_triangles};

        const GpuBuffer ssbos[] = {
            scene_data.mesh_refs,
            scene_data.materials,
            scene_data.instances,
            scene_data.light_triangles,
        };
        RtOutput   rt_output  {ctx, swapchain.extent, tlas.handle, ssbos, tex_views, tex_sampler};
        RtPipeline rt_pipeline{ctx, rt_output.descriptor_set_layout};

        Denoiser denoiser;
        denoiser.setup(swapchain.extent.width, swapchain.extent.height,
            rt_output.color_staging_ptr, rt_output.albedo_staging_ptr,
            rt_output.normal_staging_ptr, rt_output.output_staging_ptr);
        app.denoiser = &denoiser;

        // ================================================================
        // Frame loop
        // ================================================================

        uint32_t frame_index = 0;
        double   prev_time   = glfwGetTime();

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            double now = glfwGetTime();
            float  dt  = static_cast<float>(now - prev_time);
            prev_time  = now;
            camera.tick(dt);

            int fw, fh;
            glfwGetFramebufferSize(window, &fw, &fh);
            if (fw == 0 || fh == 0) continue;

            if (app.resize_needed) {
                app.resize_needed = false;
                vkDeviceWaitIdle(ctx.device.device);
                swapchain.recreate(ctx, static_cast<uint32_t>(fw), static_cast<uint32_t>(fh));
                rt_output.recreate(ctx, swapchain.extent, tlas.handle);
                denoiser.setup(swapchain.extent.width, swapchain.extent.height,
                    rt_output.color_staging_ptr, rt_output.albedo_staging_ptr,
                    rt_output.normal_staging_ptr, rt_output.output_staging_ptr);
                frame_index = 0;
                continue;
            }

            if (camera.consume_moved()) frame_index = 0;

            if (denoiser.enabled) {
                // ======================================================
                // DENOISED PATH: 3 phases, GPU stall between 1 and 2
                // ======================================================

                // Phase 1: trace rays + copy images to host-visible staging
                {
                    VkCommandBuffer cmd1 = ctx.begin_one_shot();

                    vkCmdBindPipeline(cmd1, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline.pipeline);
                    RtCameraPush push   = camera.rt_push(swapchain.extent);
                    push.frame_index    = frame_index++;
                    push.num_light_tris = scene_data.light_count;
                    vkCmdPushConstants(cmd1, rt_pipeline.layout,
                                       VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(push), &push);
                    vkCmdBindDescriptorSets(cmd1, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                            rt_pipeline.layout, 0, 1, &rt_output.descriptor_set, 0, nullptr);

                    ctx.pfn_vkCmdTraceRaysKHR(cmd1,
                        &rt_pipeline.raygen_region,
                        &rt_pipeline.miss_region,
                        &rt_pipeline.hit_region,
                        &rt_pipeline.callable_region,
                        swapchain.extent.width, swapchain.extent.height, 1);

                    VkImageMemoryBarrier readback_barriers[3]{};
                    auto setup_rb = [](VkImageMemoryBarrier& b, VkImage img) {
                        b.sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                        b.srcAccessMask    = VK_ACCESS_SHADER_WRITE_BIT;
                        b.dstAccessMask    = VK_ACCESS_TRANSFER_READ_BIT;
                        b.oldLayout        = VK_IMAGE_LAYOUT_GENERAL;
                        b.newLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                        b.image            = img;
                        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    };
                    setup_rb(readback_barriers[0], rt_output.image);
                    setup_rb(readback_barriers[1], rt_output.albedo_image);
                    setup_rb(readback_barriers[2], rt_output.normal_image);
                    vkCmdPipelineBarrier(cmd1,
                        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 0, nullptr, 0, nullptr, 3, readback_barriers);

                    VkBufferImageCopy region{};
                    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    region.imageExtent      = {swapchain.extent.width, swapchain.extent.height, 1};
                    vkCmdCopyImageToBuffer(cmd1, rt_output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        rt_output.color_staging_buf,  1, &region);
                    vkCmdCopyImageToBuffer(cmd1, rt_output.albedo_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        rt_output.albedo_staging_buf, 1, &region);
                    vkCmdCopyImageToBuffer(cmd1, rt_output.normal_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        rt_output.normal_staging_buf, 1, &region);

                    for (auto& b : readback_barriers) {
                        b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                        b.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                        b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                        b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
                    }
                    vkCmdPipelineBarrier(cmd1,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                        0, 0, nullptr, 0, nullptr, 3, readback_barriers);

                    ctx.end_one_shot(cmd1);  // submits + vkQueueWaitIdle
                }

                // Phase 2: OIDN (CPU)
                vmaInvalidateAllocation(ctx.allocator, rt_output.color_staging_alloc,  0, VK_WHOLE_SIZE);
                vmaInvalidateAllocation(ctx.allocator, rt_output.albedo_staging_alloc, 0, VK_WHOLE_SIZE);
                vmaInvalidateAllocation(ctx.allocator, rt_output.normal_staging_alloc, 0, VK_WHOLE_SIZE);
                denoiser.execute();
                vmaFlushAllocation(ctx.allocator, rt_output.output_staging_alloc, 0, VK_WHOLE_SIZE);

                // Phase 3: upload denoised result and present
                auto frame_opt = frame_sync.acquire(swapchain.handle);
                if (!frame_opt) { app.resize_needed = true; continue; }
                auto [image_index, cmd] = *frame_opt;

                VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS)
                    throw std::runtime_error("Failed to begin command buffer");

                image_barrier(cmd, rt_output.display_image,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

                VkBufferImageCopy region{};
                region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.imageExtent      = {swapchain.extent.width, swapchain.extent.height, 1};
                vkCmdCopyBufferToImage(cmd, rt_output.output_staging_buf, rt_output.display_image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

                image_barrier(cmd, rt_output.display_image,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

                image_barrier(cmd, swapchain.images[image_index],
                    0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

                {
                    VkImageBlit blit{};
                    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    blit.srcOffsets[1]  = {(int32_t)swapchain.extent.width, (int32_t)swapchain.extent.height, 1};
                    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    blit.dstOffsets[1]  = {(int32_t)swapchain.extent.width, (int32_t)swapchain.extent.height, 1};
                    vkCmdBlitImage(cmd,
                        rt_output.display_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        swapchain.images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &blit, VK_FILTER_NEAREST);
                }

                image_barrier(cmd, swapchain.images[image_index],
                    VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

                if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
                    throw std::runtime_error("Failed to end command buffer");

                VkResult present_result = frame_sync.submit_and_present(
                    ctx.graphics_queue, ctx.present_queue, swapchain.handle);
                if (present_result == VK_ERROR_OUT_OF_DATE_KHR ||
                    present_result == VK_SUBOPTIMAL_KHR)
                    app.resize_needed = true;

            } else {
                // ======================================================
                // PASSTHROUGH PATH: original pipelined trace + blit
                // ======================================================
                auto frame_opt = frame_sync.acquire(swapchain.handle);
                if (!frame_opt) { app.resize_needed = true; continue; }
                auto [image_index, cmd] = *frame_opt;

                VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS)
                    throw std::runtime_error("Failed to begin command buffer");

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline.pipeline);
                RtCameraPush push   = camera.rt_push(swapchain.extent);
                push.frame_index    = frame_index++;
                push.num_light_tris = scene_data.light_count;
                vkCmdPushConstants(cmd, rt_pipeline.layout,
                                   VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(push), &push);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                        rt_pipeline.layout, 0, 1, &rt_output.descriptor_set, 0, nullptr);

                ctx.pfn_vkCmdTraceRaysKHR(cmd,
                    &rt_pipeline.raygen_region,
                    &rt_pipeline.miss_region,
                    &rt_pipeline.hit_region,
                    &rt_pipeline.callable_region,
                    swapchain.extent.width, swapchain.extent.height, 1);

                image_barrier(cmd, rt_output.image,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT);

                image_barrier(cmd, swapchain.images[image_index],
                    0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

                {
                    VkImageBlit blit{};
                    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    blit.srcOffsets[1]  = {(int32_t)swapchain.extent.width, (int32_t)swapchain.extent.height, 1};
                    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    blit.dstOffsets[1]  = {(int32_t)swapchain.extent.width, (int32_t)swapchain.extent.height, 1};
                    vkCmdBlitImage(cmd,
                        rt_output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        swapchain.images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &blit, VK_FILTER_NEAREST);
                }

                image_barrier(cmd, swapchain.images[image_index],
                    VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

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
        }

        vkDeviceWaitIdle(ctx.device.device);
        if (gltf) gltf->destroy(ctx.device.device, ctx.allocator);

    }  // all Vulkan objects destroyed before glfwDestroyWindow

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
