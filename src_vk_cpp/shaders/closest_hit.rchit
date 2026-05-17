#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

// Barycentric coordinates of the hit point (u, v); third component = 1 - u - v.
hitAttributeEXT vec2 bary;

void main() {
    float b0 = 1.0 - bary.x - bary.y;
    payload  = vec3(b0, bary.x, bary.y);
}
