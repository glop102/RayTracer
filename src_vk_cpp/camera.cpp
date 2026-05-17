#include "camera.h"
#include "vk_context.h"

#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <stdexcept>

Camera::Camera(VkContext& ctx) {
    device    = ctx.device.device;
    allocator = ctx.allocator;

    // ------------------------------------------------------------------ Descriptor set layout
    VkDescriptorSetLayoutBinding ubo_binding{};
    ubo_binding.binding         = 0;
    ubo_binding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ubo_binding.descriptorCount = 1;
    ubo_binding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layout_ci{};
    layout_ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_ci.bindingCount = 1;
    layout_ci.pBindings    = &ubo_binding;
    if (vkCreateDescriptorSetLayout(device, &layout_ci, nullptr, &descriptor_set_layout) != VK_SUCCESS)
        throw std::runtime_error("Camera descriptor set layout creation failed");

    // ------------------------------------------------------------------ UBO buffer (persistently mapped)
    VkBufferCreateInfo buf_ci{};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size  = sizeof(glm::mat4);
    buf_ci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_ci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo alloc_info{};
    if (vmaCreateBuffer(allocator, &buf_ci, &alloc_ci, &buf, &alloc, &alloc_info) != VK_SUCCESS)
        throw std::runtime_error("Camera UBO buffer creation failed");
    mapped = alloc_info.pMappedData;

    // ------------------------------------------------------------------ Descriptor pool + set
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1};
    VkDescriptorPoolCreateInfo pool_ci{};
    pool_ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets       = 1;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes    = &pool_size;
    if (vkCreateDescriptorPool(device, &pool_ci, nullptr, &descriptor_pool) != VK_SUCCESS)
        throw std::runtime_error("Camera descriptor pool creation failed");

    VkDescriptorSetAllocateInfo set_ai{};
    set_ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_ai.descriptorPool     = descriptor_pool;
    set_ai.descriptorSetCount = 1;
    set_ai.pSetLayouts        = &descriptor_set_layout;
    if (vkAllocateDescriptorSets(device, &set_ai, &descriptor_set) != VK_SUCCESS)
        throw std::runtime_error("Camera descriptor set allocation failed");

    VkDescriptorBufferInfo buf_info{buf, 0, sizeof(glm::mat4)};
    VkWriteDescriptorSet   write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = descriptor_set;
    write.dstBinding      = 0;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo     = &buf_info;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

Camera::~Camera() {
    vkDestroyDescriptorPool     (device, descriptor_pool,       nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    vmaDestroyBuffer(allocator, buf, alloc);
}

void Camera::on_mouse_button(int button, int action) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    dragging = (action == GLFW_PRESS);
}

void Camera::on_cursor_pos(double x, double y) {
    if (dragging) {
        theta -= static_cast<float>(x - last_x) * 0.005f;
        phi    = std::clamp(phi + static_cast<float>(y - last_y) * 0.005f,
                            -std::numbers::pi_v<float> / 2.0f + 0.01f,
                             std::numbers::pi_v<float> / 2.0f - 0.01f);
        moved = true;
    }
    last_x = x;
    last_y = y;
}

void Camera::on_scroll(double dy) {
    radius = std::clamp(radius * (dy > 0 ? 0.92f : 1.08f), 0.05f, 5.0f);
    moved = true;
}

bool Camera::consume_moved() {
    bool m = moved;
    moved = false;
    return m;
}

RtCameraPush Camera::rt_push(VkExtent2D extent) const {
    float eye_x = target.x + radius * std::cos(phi) * std::sin(theta);
    float eye_y = target.y + radius * std::sin(phi);
    float eye_z = target.z + radius * std::cos(phi) * std::cos(theta);
    glm::vec3 origin{eye_x, eye_y, eye_z};

    glm::vec3 w = glm::normalize(origin - target);
    glm::vec3 u = glm::normalize(glm::cross(glm::vec3{0.0f, 1.0f, 0.0f}, w));
    glm::vec3 v = glm::cross(w, u);

    float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    float half_h = std::tan(glm::radians(45.0f) / 2.0f);
    float half_w = aspect * half_h;

    RtCameraPush push{};
    push.origin     = glm::vec4(origin, 0.0f);
    push.lower_left = glm::vec4(origin - half_w * u - half_h * v - w, 0.0f);
    push.horizontal = glm::vec4(2.0f * half_w * u, 0.0f);
    push.vertical   = glm::vec4(2.0f * half_h * v, 0.0f);
    return push;
}

void Camera::update(VkExtent2D extent) {
    float eye_x = target.x + radius * std::cos(phi) * std::sin(theta);
    float eye_y = target.y + radius * std::sin(phi);
    float eye_z = target.z + radius * std::cos(phi) * std::cos(theta);

    glm::mat4 view = glm::lookAt(glm::vec3(eye_x, eye_y, eye_z), target,
                                 glm::vec3(0.0f, 1.0f, 0.0f));
    float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.01f, 10.0f);
    proj[1][1] *= -1; // Vulkan NDC Y is inverted relative to GLM default

    glm::mat4 mvp = proj * view;
    std::memcpy(mapped, &mvp, sizeof(mvp));
}
