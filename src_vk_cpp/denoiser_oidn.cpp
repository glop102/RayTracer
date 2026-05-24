#include "denoiser_oidn.h"
#include "vk_context.h"
#include <stdexcept>
#include <cstdio>

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

OidnDenoiser::ExportBuf OidnDenoiser::make_export_buf(VkBufferUsageFlags usage) const {
    ExportBuf e;
    size_t buf_size = (size_t)w_ * h_ * 16;  // RGBA32F

    VkExternalMemoryBufferCreateInfo ext_buf_ci{};
    ext_buf_ci.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext_buf_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

    VkBufferCreateInfo buf_ci{};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.pNext = &ext_buf_ci;
    buf_ci.size  = (VkDeviceSize)buf_size;
    buf_ci.usage = usage;
    if (vkCreateBuffer(vk_device_, &buf_ci, nullptr, &e.buf) != VK_SUCCESS)
        throw std::runtime_error("OIDN export buffer creation failed");

    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(vk_device_, e.buf, &mem_req);

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);
    uint32_t mem_type_idx = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((mem_req.memoryTypeBits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            mem_type_idx = i;
            break;
        }
    }
    if (mem_type_idx == UINT32_MAX)
        throw std::runtime_error("No device-local memory type for OIDN export buffer");

    VkExportMemoryAllocateInfo export_ai{};
    export_ai.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    export_ai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

    VkMemoryAllocateInfo alloc_ai{};
    alloc_ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_ai.pNext           = &export_ai;
    alloc_ai.allocationSize  = mem_req.size;
    alloc_ai.memoryTypeIndex = mem_type_idx;
    if (vkAllocateMemory(vk_device_, &alloc_ai, nullptr, &e.mem) != VK_SUCCESS)
        throw std::runtime_error("OIDN export memory allocation failed");

    vkBindBufferMemory(vk_device_, e.buf, e.mem, 0);

    VkMemoryGetFdInfoKHR fd_info{};
    fd_info.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fd_info.memory     = e.mem;
    fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    int fd = -1;
    if (pfn_get_fd_(vk_device_, &fd_info, &fd) != VK_SUCCESS)
        throw std::runtime_error("vkGetMemoryFdKHR failed");

    // OIDN takes ownership of the fd via hipImportExternalMemory.
    e.oidn_buf = device_.newBuffer(oidn::ExternalMemoryTypeFlag::OpaqueFD, fd, buf_size);
    return e;
}

void OidnDenoiser::destroy_resources() {
    // Release OIDN references before freeing the underlying Vulkan memory.
    color_buf_.oidn_buf  = {};
    albedo_buf_.oidn_buf = {};
    normal_buf_.oidn_buf = {};
    output_buf_.oidn_buf = {};

    auto destroy_buf = [this](ExportBuf& e) {
        if (e.buf != VK_NULL_HANDLE) vkDestroyBuffer(vk_device_, e.buf, nullptr);
        if (e.mem != VK_NULL_HANDLE) vkFreeMemory(vk_device_, e.mem, nullptr);
        e = {};
    };
    destroy_buf(color_buf_);
    destroy_buf(albedo_buf_);
    destroy_buf(normal_buf_);
    destroy_buf(output_buf_);

    if (display_image_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, display_image_, display_alloc_);
        display_image_ = VK_NULL_HANDLE;
    }
}

void OidnDenoiser::setup(VkContext& ctx, uint32_t w, uint32_t h,
                          VkImage color, VkImage albedo, VkImage normal) {
    vk_device_       = ctx.device.device;
    physical_device_ = ctx.physical_device.physical_device;
    allocator_       = ctx.allocator;
    pfn_get_fd_      = ctx.pfn_vkGetMemoryFdKHR;
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

    color_buf_  = make_export_buf(VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    albedo_buf_ = make_export_buf(VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    normal_buf_ = make_export_buf(VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    output_buf_ = make_export_buf(VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    filter_ = device_.newFilter("RT");
    // pixelByteStride=16: RGBA32F layout — read only RGB, skip A.
    filter_.setImage("color",  color_buf_.oidn_buf,  oidn::Format::Float3, w, h, 0, 16);
    filter_.setImage("albedo", albedo_buf_.oidn_buf, oidn::Format::Float3, w, h, 0, 16);
    filter_.setImage("normal", normal_buf_.oidn_buf, oidn::Format::Float3, w, h, 0, 16);
    filter_.setImage("output", output_buf_.oidn_buf, oidn::Format::Float3, w, h, 0, 16);
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
    vkCmdCopyImageToBuffer(cmd, color_image_,  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, color_buf_.buf,  1, &region);
    vkCmdCopyImageToBuffer(cmd, albedo_image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, albedo_buf_.buf, 1, &region);
    vkCmdCopyImageToBuffer(cmd, normal_image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, normal_buf_.buf, 1, &region);

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
    filter_.execute();
    const char* err;
    if (device_.getError(err) != oidn::Error::None)
        fprintf(stderr, "OIDN: %s\n", err);
}

void OidnDenoiser::record_post(VkCommandBuffer cmd) {
    img_barrier(cmd, display_image_,
        0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {w_, h_, 1};
    vkCmdCopyBufferToImage(cmd, output_buf_.buf, display_image_,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    img_barrier(cmd, display_image_,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
}
