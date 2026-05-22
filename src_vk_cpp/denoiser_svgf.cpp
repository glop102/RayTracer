#include "denoiser_svgf.h"
#include "vk_context.h"
#include "shader_compiler.h"
#include <stdexcept>
#include <algorithm>

static constexpr VkFormat IMG_FMT = VK_FORMAT_R32G32B32A32_SFLOAT;

static void barrier(VkCommandBuffer cmd, VkImage image,
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

SvgfDenoiser::~SvgfDenoiser() {
    destroy_images();
    if (pipeline_)    vkDestroyPipeline(device_, pipeline_, nullptr);
    if (pipe_layout_) vkDestroyPipelineLayout(device_, pipe_layout_, nullptr);
    if (dsl_)         vkDestroyDescriptorSetLayout(device_, dsl_, nullptr);
}

SvgfDenoiser::FilterBuf SvgfDenoiser::make_filter_buf() const {
    FilterBuf fb;
    VkImageCreateInfo ci{};
    ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType     = VK_IMAGE_TYPE_2D;
    ci.format        = IMG_FMT;
    ci.extent        = {w_, h_, 1};
    ci.mipLevels     = 1;
    ci.arrayLayers   = 1;
    ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ci.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    ai.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    if (vmaCreateImage(allocator_, &ci, &ai, &fb.image, &fb.alloc, nullptr) != VK_SUCCESS)
        throw std::runtime_error("SVGF filter buffer creation failed");
    fb.view = make_view(fb.image);
    return fb;
}

VkImageView SvgfDenoiser::make_view(VkImage image) const {
    VkImageViewCreateInfo vi{};
    vi.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image            = image;
    vi.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    vi.format           = IMG_FMT;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view;
    if (vkCreateImageView(device_, &vi, nullptr, &view) != VK_SUCCESS)
        throw std::runtime_error("SVGF image view creation failed");
    return view;
}

void SvgfDenoiser::destroy_images() {
    if (pool_) {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
        pool_ = VK_NULL_HANDLE;
    }
    for (auto* v : {&color_view_, &albedo_view_, &normal_view_}) {
        if (*v) { vkDestroyImageView(device_, *v, nullptr); *v = VK_NULL_HANDLE; }
    }
    for (auto& fb : bufs_) {
        if (fb.view)  { vkDestroyImageView(device_, fb.view, nullptr); fb.view = VK_NULL_HANDLE; }
        if (fb.image) { vmaDestroyImage(allocator_, fb.image, fb.alloc); fb.image = VK_NULL_HANDLE; }
    }
}

void SvgfDenoiser::create_pipeline(VkContext& ctx) {
    auto shader_dir = find_shader_dir();
    auto spv = compile_glsl(read_file(shader_dir / "svgf_atrous.comp"),
                            "svgf_atrous.comp", shaderc_glsl_compute_shader);
    VkShaderModule mod = make_module(ctx.device.device, spv);

    VkDescriptorSetLayoutBinding bindings[4]{};
    for (int i = 0; i < 4; i++) {
        bindings[i] = {(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                       1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }
    VkDescriptorSetLayoutCreateInfo dsl_ci{};
    dsl_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = 4;
    dsl_ci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(ctx.device.device, &dsl_ci, nullptr, &dsl_) != VK_SUCCESS)
        throw std::runtime_error("SVGF DSL creation failed");

    VkPushConstantRange pc_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int)};
    VkPipelineLayoutCreateInfo layout_ci{};
    layout_ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount         = 1;
    layout_ci.pSetLayouts            = &dsl_;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges    = &pc_range;
    if (vkCreatePipelineLayout(ctx.device.device, &layout_ci, nullptr, &pipe_layout_) != VK_SUCCESS)
        throw std::runtime_error("SVGF pipeline layout creation failed");

    VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr};
    VkComputePipelineCreateInfo pipe_ci{};
    pipe_ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipe_ci.stage  = stage;
    pipe_ci.layout = pipe_layout_;
    if (vkCreateComputePipelines(ctx.device.device, VK_NULL_HANDLE, 1, &pipe_ci, nullptr, &pipeline_) != VK_SUCCESS)
        throw std::runtime_error("SVGF pipeline creation failed");

    vkDestroyShaderModule(ctx.device.device, mod, nullptr);
}

void SvgfDenoiser::setup(VkContext& ctx, uint32_t w, uint32_t h,
                          VkImage color, VkImage albedo, VkImage normal) {
    device_    = ctx.device.device;
    allocator_ = ctx.allocator;
    w_ = w; h_ = h;
    color_image_  = color;
    albedo_image_ = albedo;
    normal_image_ = normal;

    if (!pipeline_) create_pipeline(ctx);

    destroy_images();

    color_view_  = make_view(color_image_);
    albedo_view_ = make_view(albedo_image_);
    normal_view_ = make_view(normal_image_);
    bufs_[0] = make_filter_buf();
    bufs_[1] = make_filter_buf();

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 * NUM_PASSES};
    VkDescriptorPoolCreateInfo pool_ci{};
    pool_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets       = NUM_PASSES;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(device_, &pool_ci, nullptr, &pool_) != VK_SUCCESS)
        throw std::runtime_error("SVGF descriptor pool creation failed");

    VkDescriptorSetLayout layouts[NUM_PASSES];
    std::fill(layouts, layouts + NUM_PASSES, dsl_);
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool     = pool_;
    alloc_info.descriptorSetCount = NUM_PASSES;
    alloc_info.pSetLayouts        = layouts;
    if (vkAllocateDescriptorSets(device_, &alloc_info, dsets_) != VK_SUCCESS)
        throw std::runtime_error("SVGF descriptor set allocation failed");

    // Pass i reads from in_views[i] and writes to out_views[i]:
    //   0: color → bufs_[0]
    //   1: bufs_[0] → bufs_[1]
    //   2: bufs_[1] → bufs_[0]
    //   3: bufs_[0] → bufs_[1]   final output: bufs_[1]
    const VkImageView in_views[NUM_PASSES]  = {color_view_,   bufs_[0].view, bufs_[1].view, bufs_[0].view};
    const VkImageView out_views[NUM_PASSES] = {bufs_[0].view, bufs_[1].view, bufs_[0].view, bufs_[1].view};

    for (int i = 0; i < NUM_PASSES; i++) {
        VkDescriptorImageInfo img_infos[4] = {
            {VK_NULL_HANDLE, in_views[i],   VK_IMAGE_LAYOUT_GENERAL},
            {VK_NULL_HANDLE, normal_view_,  VK_IMAGE_LAYOUT_GENERAL},
            {VK_NULL_HANDLE, out_views[i],  VK_IMAGE_LAYOUT_GENERAL},
            {VK_NULL_HANDLE, albedo_view_,  VK_IMAGE_LAYOUT_GENERAL},
        };
        VkWriteDescriptorSet writes[4]{};
        for (int b = 0; b < 4; b++) {
            writes[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet          = dsets_[i];
            writes[b].dstBinding      = (uint32_t)b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[b].pImageInfo      = &img_infos[b];
        }
        vkUpdateDescriptorSets(device_, 4, writes, 0, nullptr);
    }
}

VkImage SvgfDenoiser::output_image() const {
    return bufs_[(NUM_PASSES - 1) % 2].image;  // bufs_[1] for 4 passes
}

void SvgfDenoiser::record_pre(VkCommandBuffer cmd) {
    // Transition ping-pong buffers to GENERAL, discarding previous contents.
    for (auto& fb : bufs_) {
        barrier(cmd, fb.image,
            0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    // Memory dependency: raytrace SHADER_WRITE → compute SHADER_READ on all G-buffers.
    VkImageMemoryBarrier rt_barriers[3]{};
    const VkImage rt_images[3] = {color_image_, albedo_image_, normal_image_};
    for (int i = 0; i < 3; i++) {
        rt_barriers[i].sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        rt_barriers[i].srcAccessMask    = VK_ACCESS_SHADER_WRITE_BIT;
        rt_barriers[i].dstAccessMask    = VK_ACCESS_SHADER_READ_BIT;
        rt_barriers[i].oldLayout        = VK_IMAGE_LAYOUT_GENERAL;
        rt_barriers[i].newLayout        = VK_IMAGE_LAYOUT_GENERAL;
        rt_barriers[i].image            = rt_images[i];
        rt_barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 3, rt_barriers);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);

    const uint32_t gx = (w_ + 7) / 8;
    const uint32_t gy = (h_ + 7) / 8;
    const int step_widths[NUM_PASSES] = {1, 2, 4, 8};

    for (int i = 0; i < NUM_PASSES; i++) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe_layout_, 0, 1, &dsets_[i], 0, nullptr);
        vkCmdPushConstants(cmd, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(int), &step_widths[i]);
        vkCmdDispatch(cmd, gx, gy, 1);

        if (i < NUM_PASSES - 1) {
            // The output of pass i feeds into pass i+1 as input.
            barrier(cmd, bufs_[i % 2].image,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
    }

    // Transition final output to TRANSFER_SRC_OPTIMAL for the swapchain blit.
    barrier(cmd, bufs_[(NUM_PASSES - 1) % 2].image,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    // Restore G-buffers so raygen can write them next frame.
    for (int i = 0; i < 3; i++) {
        rt_barriers[i].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        rt_barriers[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    }
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 3, rt_barriers);
}
