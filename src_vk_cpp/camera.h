#pragma once
#include <vulkan/vulkan.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <cstdint>

struct GLFWwindow;

// Push constant layout for the raygen shader.
// Four vec4s = 64 bytes, fits in the guaranteed 128-byte minimum.
struct RtCameraPush {
    glm::vec4 origin;       // xyz = eye position
    glm::vec4 lower_left;   // xyz = lower-left corner of virtual screen at distance 1
    glm::vec4 horizontal;   // xyz = full horizontal span of virtual screen
    glm::vec4 vertical;     // xyz = full vertical span of virtual screen
    uint32_t  frame_index;     // 0 = first frame with current camera; drives accumulation weight
    uint32_t  num_light_tris;  // number of emissive triangles in the light list SSBO
    float     _pad[2];
};

// First-person camera with mouse-look and WASD/EQ movement.
// Click in the window to capture the mouse; press ESC to release.
struct Camera {
    Camera() = default;

    void set_window(GLFWwindow* w) { window = w; }
    void reset_pose(glm::vec3 p, float y, float pi) { pos = p; yaw = y; pitch = pi; moved = true; }

    // GLFW callback handlers — wire these up via the window user pointer.
    void on_mouse_button(int button, int action);
    void on_cursor_pos(double x, double y);

    // Poll keyboard and advance position; call once per frame with elapsed seconds.
    void tick(float dt);

    // Build push constant data for the raygen shader.
    RtCameraPush rt_push(VkExtent2D extent) const;

    // Returns true (and clears the flag) if the camera moved since the last call.
    bool consume_moved();

    bool moved = false;

    Camera(const Camera&)            = delete;
    Camera& operator=(const Camera&) = delete;

private:
    GLFWwindow* window = nullptr;

    glm::vec3 pos   = {0.0f, 0.20f, 0.8f};
    float     yaw   = 3.14159265f;  // looking in -Z (toward the scene)
    float     pitch = -0.1f;        // slightly downward

    bool   captured  = false;
    double last_x    = 0.0;
    double last_y    = 0.0;
    bool   first_pos = true;

    static constexpr float MOVE_SPEED       = 0.5f;   // m/s
    static constexpr float LOOK_SENSITIVITY = 0.002f; // rad/pixel
};
