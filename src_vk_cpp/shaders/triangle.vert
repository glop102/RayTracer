#version 450

layout(location = 0) in vec3 pos;

layout(push_constant) uniform PC {
    mat4 mvp;
} pc;

layout(location = 0) out vec3 world_pos;

void main() {
    gl_Position = pc.mvp * vec4(pos, 1.0);
    world_pos   = pos;
}
