#pragma once
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

struct VkContext;

// Push constant layout for the raygen shader.
// Four vec4s = 64 bytes, fits in the guaranteed 128-byte minimum.
struct RtCameraPush {
    glm::vec4 origin;      // xyz = eye position
    glm::vec4 lower_left;  // xyz = lower-left corner of virtual screen at distance 1
    glm::vec4 horizontal;  // xyz = full horizontal span of virtual screen
    glm::vec4 vertical;    // xyz = full vertical span of virtual screen
};

// Owns the camera UBO, its descriptor set, and orbit input state.
// Construct before Pipeline — pipeline layout needs descriptor_set_layout.
struct Camera {
    // Pipeline layout needs this; created in constructor.
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet       descriptor_set        = VK_NULL_HANDLE;

    Camera(VkContext& ctx);
    ~Camera();

    // GLFW callback handlers — wire these up via the window user pointer.
    void on_mouse_button(int button, int action);
    void on_cursor_pos(double x, double y);
    void on_scroll(double dy);

    // Recompute MVP from current orbit state and write it into the UBO.
    void update(VkExtent2D extent);

    // Build push constant data for the raygen shader from current orbit state.
    RtCameraPush rt_push(VkExtent2D extent) const;

    Camera(const Camera&)            = delete;
    Camera& operator=(const Camera&) = delete;

private:
    VkDevice      device    = VK_NULL_HANDLE;
    VmaAllocator  allocator = VK_NULL_HANDLE;

    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkBuffer         buf             = VK_NULL_HANDLE;
    VmaAllocation    alloc           = {};
    void*            mapped          = nullptr;

    // Spherical orbit state
    float     theta  =  0.0f;   // azimuth
    float     phi    =  0.2f;   // elevation
    float     radius =  0.25f;
    glm::vec3 target = {0.0f, 0.10f, 0.0f};

    // Mouse drag tracking
    bool   dragging = false;
    double last_x   = 0.0;
    double last_y   = 0.0;
};
