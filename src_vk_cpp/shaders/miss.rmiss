#version 460
#extension GL_EXT_ray_tracing : require

struct HitResult {
    vec3  position;
    vec3  normal;
    vec3  diffuse;
    vec3  specular;
    vec3  emissive;
    vec3  absorption;
    float roughness;
    float ior;
    int   hit;
};
layout(location = 0) rayPayloadInEXT HitResult payload;

void main() {
    payload.hit      = 0;
    payload.emissive = vec3(0.0);
}
