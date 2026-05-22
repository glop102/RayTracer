#include "camera.h"

#include <GLFW/glfw3.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

void Camera::on_mouse_button(int button, int action) {
    if (!window || button != GLFW_MOUSE_BUTTON_LEFT || action != GLFW_PRESS) return;
    if (!captured) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        first_pos = true;
        captured  = true;
    }
}

void Camera::on_cursor_pos(double x, double y) {
    if (!captured) { last_x = x; last_y = y; return; }
    if (first_pos) { last_x = x; last_y = y; first_pos = false; return; }

    yaw   -= static_cast<float>(x - last_x) * LOOK_SENSITIVITY;
    pitch  = std::clamp(pitch - static_cast<float>(y - last_y) * LOOK_SENSITIVITY,
                        -std::numbers::pi_v<float> / 2.0f + 0.01f,
                         std::numbers::pi_v<float> / 2.0f - 0.01f);
    last_x = x;
    last_y = y;
    moved  = true;
}

void Camera::tick(float dt) {
    if (!window) return;

    if (captured && glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        captured = false;
    }

    float cp = std::cos(pitch), sp = std::sin(pitch);
    float cy = std::cos(yaw),   sy = std::sin(yaw);
    glm::vec3 forward = {cp * sy, sp, cp * cy};
    glm::vec3 right   = glm::normalize(glm::cross(forward, glm::vec3{0.0f, 1.0f, 0.0f}));

    glm::vec3 delta{0.0f};
    bool alt = glfwGetKey(window, GLFW_KEY_LEFT_ALT)  == GLFW_PRESS ||
               glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
    if (!alt) {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) delta += forward;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) delta -= forward;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) delta += right;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) delta -= right;
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) delta.y += 1.0f;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) delta.y -= 1.0f;
    }

    if (glm::length(delta) > 0.0f) {
        pos   += glm::normalize(delta) * (MOVE_SPEED * dt);
        moved  = true;
    }
}

bool Camera::consume_moved() {
    bool m = moved;
    moved = false;
    return m;
}

RtCameraPush Camera::rt_push(VkExtent2D extent) const {
    float cp = std::cos(pitch), sp = std::sin(pitch);
    float cy = std::cos(yaw),   sy = std::sin(yaw);
    glm::vec3 forward = {cp * sy, sp, cp * cy};
    glm::vec3 right   = glm::normalize(glm::cross(forward, glm::vec3{0.0f, 1.0f, 0.0f}));
    glm::vec3 up      = glm::cross(right, forward);

    float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    float half_h = std::tan(glm::radians(45.0f) / 2.0f);
    float half_w = aspect * half_h;

    RtCameraPush push{};
    push.origin     = glm::vec4(pos, 0.0f);
    push.lower_left = glm::vec4(pos + forward - half_w * right - half_h * up, 0.0f);
    push.horizontal = glm::vec4(2.0f * half_w * right, 0.0f);
    push.vertical   = glm::vec4(2.0f * half_h * up, 0.0f);
    return push;
}
