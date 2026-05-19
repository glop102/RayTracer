#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 1) rayPayloadInEXT int shadow_hit;

void main() {
    shadow_hit = 0; // miss = nothing between hit point and light → light visible
}
