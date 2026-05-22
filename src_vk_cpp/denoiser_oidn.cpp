#include "denoiser_oidn.h"
#include "vk_context.h"
#include <stdexcept>
#include <cstdio>
#include <cstring>

static constexpr VkFormat DISPLAY_FORMAT = VK_FORMAT_R32G32B32A32_SFLOAT;

static void img_barrier(VkCommandBuffer cmd, VkImage image,
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

OidnDenoiser::OidnDenoiser() {
    device_ = oidn::newDevice(oidn::DeviceType::HIP);
    device_.commit();
    const char* err;
    if (device_.getError(err) != oidn::Error::None) {
        fprintf(stderr, "OIDN HIP unavailable (%s), falling back to CPU\n", err);
        device_ = oidn::newDevice(oidn::DeviceType::CPU);
        device_.commit();
        if (device_.getError(err) != oidn::Error::None)
            fprintf(stderr, "OIDN CPU init failed: %s\n", err);
    }
}

OidnDenoiser::~OidnDenoiser() {
    destroy_resources();
}

OidnDenoiser::StagingBuf OidnDenoiser::make_staging(VkBufferUsageFlags usage) const {
    StagingBuf s;
    VkBufferCreateInfo buf_ci{};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size  = (VkDeviceSize)w_ * h_ * 16;  // RGBA32F
    buf_ci.usage = usage;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_ci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    if (vmaCreateBuffer(allocator_, &buf_ci, &alloc_ci, &s.buf, &s.alloc, &info) != VK_SUCCESS)
        throw std::runtime_error("OIDN staging buffer creation failed");
    s.ptr = info.pMappedData;
    return s;
}

void OidnDenoiser::destroy_resources() {
    if (color_stg_.buf  != VK_NULL_HANDLE) vmaDestroyBuffer(allocator_, color_stg_.buf,  color_stg_.alloc);
    if (albedo_stg_.buf != VK_NULL_HANDLE) vmaDestroyBuffer(allocator_, albedo_stg_.buf, albedo_stg_.alloc);
    if (normal_stg_.buf != VK_NULL_HANDLE) vmaDestroyBuffer(allocator_, normal_stg_.buf, normal_stg_.alloc);
    if (output_stg_.buf != VK_NULL_HANDLE) vmaDestroyBuffer(allocator_, output_stg_.buf, output_stg_.alloc);
    color_stg_ = albedo_stg_ = normal_stg_ = output_stg_ = {};

    if (display_image_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, display_image_, display_alloc_);
        display_image_ = VK_NULL_HANDLE;
    }

    color_oidn_buf_ = albedo_oidn_buf_ = normal_oidn_buf_ = output_oidn_buf_ = {};
}

void OidnDenoiser::setup(VkContext& ctx, uint32_t w, uint32_t h,
                          VkImage color, VkImage albedo, VkImage normal) {
    vk_device_   = ctx.device.device;
    allocator_   = ctx.allocator;
    w_ = w;  h_ = h;
    color_image_  = color;
    albedo_image_ = albedo;
    normal_image_ = normal;

    destroy_resources();

    VkImageCreateInfo img_ci{};
    img_ci.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img_ci.imageType     = VK_IMAGE_TYPE_2D;
    img_ci.format        = DISPLAY_FORMAT;
    img_ci.extent        = {w, h, 1};
    img_ci.mipLevels     = 1;
    img_ci.arrayLayers   = 1;
    img_ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    img_ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    img_ci.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo img_ai{};
    img_ai.usage = VMA_MEMORY_USAGE_AUTO;
    img_ai.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    if (vmaCreateImage(allocator_, &img_ci, &img_ai, &display_image_, &display_alloc_, nullptr) != VK_SUCCESS)
        throw std::runtime_error("OIDN display image creation failed");

    color_stg_  = make_staging(VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    albedo_stg_ = make_staging(VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    normal_stg_ = make_staging(VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    output_stg_ = make_staging(VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    // Allocate OIDN host-pinned buffers (accessible by both CPU and HIP device).
    // VMA staging buffers hold the GPU readback; we memcpy into these before executing.
    size_t buf_size = (size_t)w * h * 16;
    color_oidn_buf_  = device_.newBuffer(buf_size, oidn::Storage::Host);
    albedo_oidn_buf_ = device_.newBuffer(buf_size, oidn::Storage::Host);
    normal_oidn_buf_ = device_.newBuffer(buf_size, oidn::Storage::Host);
    output_oidn_buf_ = device_.newBuffer(buf_size, oidn::Storage::Host);

    filter_ = device_.newFilter("RT");
    // pixelByteStride=16: RGBA32F layout — read only RGB, skip A.
    filter_.setImage("color",  color_oidn_buf_,  oidn::Format::Float3, w, h, 0, 16);
    filter_.setImage("albedo", albedo_oidn_buf_, oidn::Format::Float3, w, h, 0, 16);
    filter_.setImage("normal", normal_oidn_buf_, oidn::Format::Float3, w, h, 0, 16);
    filter_.setImage("output", output_oidn_buf_, oidn::Format::Float3, w, h, 0, 16);
    filter_.set("hdr", true);
    filter_.set("quality", oidn::Quality::Balanced);
    filter_.commit();
}

void OidnDenoiser::record_pre(VkCommandBuffer cmd) {
    VkImageMemoryBarrier barriers[3]{};
    for (int i = 0; i < 3; i++) {
        barriers[i].sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[i].srcAccessMask    = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[i].dstAccessMask    = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[i].oldLayout        = VK_IMAGE_LAYOUT_GENERAL;
        barriers[i].newLayout        = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    barriers[0].image = color_image_;
    barriers[1].image = albedo_image_;
    barriers[2].image = normal_image_;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 3, barriers);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {w_, h_, 1};
    vkCmdCopyImageToBuffer(cmd, color_image_,  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, color_stg_.buf,  1, &region);
    vkCmdCopyImageToBuffer(cmd, albedo_image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, albedo_stg_.buf, 1, &region);
    vkCmdCopyImageToBuffer(cmd, normal_image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, normal_stg_.buf, 1, &region);

    for (auto& b : barriers) {
        b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
    }
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 3, barriers);
}

void OidnDenoiser::execute() {
    size_t buf_size = (size_t)w_ * h_ * 16;
    vmaInvalidateAllocation(allocator_, color_stg_.alloc,  0, VK_WHOLE_SIZE);
    vmaInvalidateAllocation(allocator_, albedo_stg_.alloc, 0, VK_WHOLE_SIZE);
    vmaInvalidateAllocation(allocator_, normal_stg_.alloc, 0, VK_WHOLE_SIZE);
    memcpy(color_oidn_buf_.getData(),  color_stg_.ptr,  buf_size);
    memcpy(albedo_oidn_buf_.getData(), albedo_stg_.ptr, buf_size);
    memcpy(normal_oidn_buf_.getData(), normal_stg_.ptr, buf_size);
    filter_.execute();
    const char* err;
    if (device_.getError(err) != oidn::Error::None)
        fprintf(stderr, "OIDN: %s\n", err);
    memcpy(output_stg_.ptr, output_oidn_buf_.getData(), buf_size);
    vmaFlushAllocation(allocator_, output_stg_.alloc, 0, VK_WHOLE_SIZE);
}

void OidnDenoiser::record_post(VkCommandBuffer cmd) {
    img_barrier(cmd, display_image_,
        0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {w_, h_, 1};
    vkCmdCopyBufferToImage(cmd, output_stg_.buf, display_image_,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    img_barrier(cmd, display_image_,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
}
