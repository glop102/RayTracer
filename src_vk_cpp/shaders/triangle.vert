#version 450

layout(location = 0) in vec3 pos;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 mvp;
} camera;

layout(location = 0) out vec3 world_pos;

void main() {
    gl_Position = camera.mvp * vec4(pos, 1.0);
    world_pos   = pos;
}
