#version 460
#extension GL_EXT_ray_tracing : require

struct HitResult {
    vec3  position;
    vec3  normal;
    vec3  diffuse;
    vec3  specular;
    vec3  emissive;
    float roughness;
    int   hit;
};
layout(location = 0) rayPayloadInEXT HitResult payload;

void main() {
    float y = normalize(gl_WorldRayDirectionEXT).y;
    vec3 sky;
    if (y > 0.0)
        sky = mix(vec3(1.0), vec3(0.4, 0.6, 0.9), y);
    else if (y > -0.5)
        sky = vec3(1.0) * (1.0 + y * 2.0);
    else
        sky = vec3(0.0);

    payload.hit      = 0;
    payload.emissive = sky;
}
