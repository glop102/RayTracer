#include "mesh.h"
#include "vk_context.h"

#include <glm/glm.hpp>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Parse an ASCII PLY file into vertex positions and triangle indices.
// Handles extra per-vertex properties beyond x,y,z (e.g. confidence, intensity).
// Polygons with more than 3 vertices are fan-triangulated.
static void load_ply(const std::string& path,
                     std::vector<glm::vec3>& verts,
                     std::vector<uint32_t>&  indices) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open PLY: " + path);

    uint32_t nv = 0, nf = 0;
    int      vprop_count = 0;

    enum { NONE, IN_VERTEX, IN_FACE } section = NONE;

    std::string line;
    std::getline(f, line); // "ply"
    while (!line.starts_with("end_header")) {
        if (line.starts_with("element vertex ")) {
            nv      = std::stoul(line.substr(strlen("element vertex ")));
            section = IN_VERTEX;
        } else if (line.starts_with("element face ")) {
            nf      = std::stoul(line.substr(strlen("element face ")));
            section = IN_FACE;
        } else if (section == IN_VERTEX &&
                   (line.starts_with("property float ") ||
                    line.starts_with("property double "))) {
            vprop_count++;
        }
        std::getline(f, line);
    }

    int skip = vprop_count - 3; // extra floats after x, y, z
    if (skip < 0) skip = 0;

    verts.reserve(nv);
    for (uint32_t i = 0; i < nv; i++) {
        float x, y, z, dummy;
        f >> x >> y >> z;
        for (int s = 0; s < skip; s++) f >> dummy;
        verts.push_back({x, y, z});
        std::getline(f, line); // consume trailing newline
    }

    indices.reserve(nf * 3);
    for (uint32_t i = 0; i < nf; i++) {
        int count;
        f >> count;
        uint32_t a, b, c;
        f >> a >> b >> c;
        indices.push_back(a);
        indices.push_back(b);
        indices.push_back(c);
        // fan-triangulate any polygon with more than 3 vertices
        for (int e = 3; e < count; e++) {
            uint32_t d;
            f >> d;
            indices.push_back(a);
            indices.push_back(c);
            indices.push_back(d);
            c = d;
        }
        std::getline(f, line);
    }
}

static VkCommandBuffer begin_one_shot(VkContext& ctx) {
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool        = ctx.command_pool;
    alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;

    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(ctx.device.device, &alloc, &cmd) != VK_SUCCESS)
        throw std::runtime_error("One-shot command buffer allocation failed");

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    return cmd;
}

static void end_one_shot(VkContext& ctx, VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{};
    submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd;
    vkQueueSubmit(ctx.graphics_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.graphics_queue);

    vkFreeCommandBuffers(ctx.device.device, ctx.command_pool, 1, &cmd);
}

static void upload_buffer(VkContext& ctx,
                          VkBufferUsageFlags usage,
                          const void* data, VkDeviceSize size,
                          VkBuffer& out_buf, VmaAllocation& out_alloc) {
    // CPU-visible staging buffer
    VkBufferCreateInfo stg_ci{};
    stg_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stg_ci.size  = size;
    stg_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stg_ai{};
    stg_ai.usage = VMA_MEMORY_USAGE_AUTO;
    stg_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VkBuffer      stg_buf;
    VmaAllocation stg_alloc;
    if (vmaCreateBuffer(ctx.allocator, &stg_ci, &stg_ai,
                        &stg_buf, &stg_alloc, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Staging buffer creation failed");

    void* mapped;
    vmaMapMemory(ctx.allocator, stg_alloc, &mapped);
    std::memcpy(mapped, data, size);
    vmaUnmapMemory(ctx.allocator, stg_alloc);

    // Device-local destination buffer
    VkBufferCreateInfo buf_ci{};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size  = size;
    buf_ci.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo buf_ai{};
    buf_ai.usage = VMA_MEMORY_USAGE_AUTO;
    buf_ai.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    if (vmaCreateBuffer(ctx.allocator, &buf_ci, &buf_ai,
                        &out_buf, &out_alloc, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Device buffer creation failed");

    VkCommandBuffer cmd = begin_one_shot(ctx);
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cmd, stg_buf, out_buf, 1, &region);
    end_one_shot(ctx, cmd);

    vmaDestroyBuffer(ctx.allocator, stg_buf, stg_alloc);
}

Mesh::Mesh(VkContext& ctx, const std::string& ply_path) {
    device    = ctx.device.device;
    allocator = ctx.allocator;

    std::vector<glm::vec3> verts;
    std::vector<uint32_t>  inds;
    load_ply(ply_path, verts, inds);
    index_count = static_cast<uint32_t>(inds.size());

    upload_buffer(ctx, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                  verts.data(), verts.size() * sizeof(glm::vec3),
                  vertex_buf, vertex_alloc);

    upload_buffer(ctx, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                  inds.data(), inds.size() * sizeof(uint32_t),
                  index_buf, index_alloc);
}

Mesh::~Mesh() {
    vmaDestroyBuffer(allocator, vertex_buf, vertex_alloc);
    vmaDestroyBuffer(allocator, index_buf,  index_alloc);
}
