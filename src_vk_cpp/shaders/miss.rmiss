#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
    // Sky gradient: white at horizon, blue at zenith.
    float t  = 0.5 * (normalize(gl_WorldRayDirectionEXT).y + 1.0);
    payload  = mix(vec3(1.0), vec3(0.5, 0.7, 1.0), t);
}
