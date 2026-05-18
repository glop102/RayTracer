#include "mesh.h"
#include "vk_context.h"

#include <glm/glm.hpp>
#include <glm/geometric.hpp>

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

// Area-weighted vertex normal accumulation — smoother than angle-weighted for
// meshes like the Stanford bunny where triangle sizes vary significantly.
static void compute_normals(const std::vector<glm::vec3>& verts,
                             const std::vector<uint32_t>&  inds,
                             std::vector<glm::vec3>&       normals) {
    normals.assign(verts.size(), glm::vec3(0.0f));
    for (size_t i = 0; i < inds.size(); i += 3) {
        uint32_t i0 = inds[i], i1 = inds[i+1], i2 = inds[i+2];
        glm::vec3 face_n = glm::cross(verts[i1] - verts[i0], verts[i2] - verts[i0]);
        normals[i0] += face_n;
        normals[i1] += face_n;
        normals[i2] += face_n;
    }
    for (auto& n : normals) {
        float len = glm::length(n);
        if (len > 1e-6f) n /= len;
    }
}

// Usage flags required for RT BLAS geometry input + rasterization vertex/index.
static constexpr VkBufferUsageFlags VERTEX_USAGE =
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

static constexpr VkBufferUsageFlags INDEX_USAGE =
    VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

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

    VkCommandBuffer cmd = ctx.begin_one_shot();
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cmd, stg_buf, out_buf, 1, &region);
    ctx.end_one_shot(cmd);

    vmaDestroyBuffer(ctx.allocator, stg_buf, stg_alloc);
}

Mesh::Mesh(VkContext& ctx, const std::string& ply_path) {
    device    = ctx.device.device;
    allocator = ctx.allocator;

    std::vector<glm::vec3> verts;
    std::vector<uint32_t>  inds;
    load_ply(ply_path, verts, inds);

    vertex_count = static_cast<uint32_t>(verts.size());
    index_count  = static_cast<uint32_t>(inds.size());

    // Compute smooth normals and interleave with positions: {x,y,z, nx,ny,nz}.
    // The BLAS reads only the position portion (stride = 6 floats).
    // The closest-hit shader reads normals at offset +3.
    std::vector<glm::vec3> normals;
    compute_normals(verts, inds, normals);

    std::vector<float> vdata;
    vdata.reserve(verts.size() * 6);
    for (size_t i = 0; i < verts.size(); i++) {
        vdata.push_back(verts[i].x);   vdata.push_back(verts[i].y);   vdata.push_back(verts[i].z);
        vdata.push_back(normals[i].x); vdata.push_back(normals[i].y); vdata.push_back(normals[i].z);
    }

    upload_buffer(ctx, VERTEX_USAGE,
                  vdata.data(), vdata.size() * sizeof(float),
                  vertex_buf, vertex_alloc);

    upload_buffer(ctx, INDEX_USAGE,
                  inds.data(), inds.size() * sizeof(uint32_t),
                  index_buf, index_alloc);

    // Device addresses needed by BLAS geometry descriptor.
    VkBufferDeviceAddressInfo addr_info{};
    addr_info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addr_info.buffer = vertex_buf;
    vertex_addr = vkGetBufferDeviceAddress(device, &addr_info);

    addr_info.buffer = index_buf;
    index_addr = vkGetBufferDeviceAddress(device, &addr_info);
}

Mesh::Mesh(VkContext& ctx, glm::vec3 mn, glm::vec3 mx) {
    device    = ctx.device.device;
    allocator = ctx.allocator;

    // 6 faces × 4 vertices = 24 vertices; 6 faces × 2 triangles × 3 = 36 indices.
    // Vertices have CCW winding when viewed from outside (outward face normals).
    struct Face { glm::vec3 v[4]; glm::vec3 n; };
    const Face faces[6] = {
        {{{mx.x,mn.y,mn.z},{mx.x,mx.y,mn.z},{mx.x,mx.y,mx.z},{mx.x,mn.y,mx.z}}, { 1, 0, 0}},
        {{{mn.x,mn.y,mx.z},{mn.x,mx.y,mx.z},{mn.x,mx.y,mn.z},{mn.x,mn.y,mn.z}}, {-1, 0, 0}},
        {{{mn.x,mx.y,mx.z},{mx.x,mx.y,mx.z},{mx.x,mx.y,mn.z},{mn.x,mx.y,mn.z}}, { 0, 1, 0}},
        {{{mn.x,mn.y,mn.z},{mx.x,mn.y,mn.z},{mx.x,mn.y,mx.z},{mn.x,mn.y,mx.z}}, { 0,-1, 0}},
        {{{mn.x,mn.y,mx.z},{mx.x,mn.y,mx.z},{mx.x,mx.y,mx.z},{mn.x,mx.y,mx.z}}, { 0, 0, 1}},
        {{{mx.x,mn.y,mn.z},{mn.x,mn.y,mn.z},{mn.x,mx.y,mn.z},{mx.x,mx.y,mn.z}}, { 0, 0,-1}},
    };

    std::vector<float>    vdata;
    std::vector<uint32_t> inds;
    vdata.reserve(24 * 6);
    inds.reserve(36);

    for (const auto& f : faces) {
        auto base = static_cast<uint32_t>(vdata.size() / 6);
        for (const auto& p : f.v) {
            vdata.push_back(p.x); vdata.push_back(p.y); vdata.push_back(p.z);
            vdata.push_back(f.n.x); vdata.push_back(f.n.y); vdata.push_back(f.n.z);
        }
        inds.push_back(base+0); inds.push_back(base+1); inds.push_back(base+2);
        inds.push_back(base+0); inds.push_back(base+2); inds.push_back(base+3);
    }

    vertex_count = static_cast<uint32_t>(vdata.size() / 6);
    index_count  = static_cast<uint32_t>(inds.size());

    upload_buffer(ctx, VERTEX_USAGE, vdata.data(), vdata.size() * sizeof(float),
                  vertex_buf, vertex_alloc);
    upload_buffer(ctx, INDEX_USAGE,  inds.data(),  inds.size()  * sizeof(uint32_t),
                  index_buf,  index_alloc);

    VkBufferDeviceAddressInfo addr_info{};
    addr_info.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addr_info.buffer = vertex_buf;
    vertex_addr = vkGetBufferDeviceAddress(device, &addr_info);

    addr_info.buffer = index_buf;
    index_addr = vkGetBufferDeviceAddress(device, &addr_info);
}

Mesh::~Mesh() {
    vmaDestroyBuffer(allocator, vertex_buf, vertex_alloc);
    vmaDestroyBuffer(allocator, index_buf,  index_alloc);
}
