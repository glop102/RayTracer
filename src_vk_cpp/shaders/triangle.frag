#version 450

layout(location = 0) in  vec3 world_pos;
layout(location = 0) out vec4 out_color;

void main() {
    vec3 normal = normalize(cross(dFdx(world_pos), dFdy(world_pos)));
    out_color = vec4(normal * 0.5 + 0.5, 1.0);
}
