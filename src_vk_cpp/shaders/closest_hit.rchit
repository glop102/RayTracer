#version 460
#extension GL_EXT_ray_tracing                            : require
#extension GL_EXT_buffer_reference                       : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_nonuniform_qualifier                   : require

#include "hit_common.glsl"

layout(set = 0, binding = 6) uniform sampler2D textures[];

void main() {
    uint         iid  = gl_InstanceCustomIndexEXT;
    InstanceData inst = instances.d[iid];
    MeshRef      mesh = mesh_refs.d[inst.mesh_idx];
    GpuMaterial  mat  = materials.d[inst.mat_idx];

    uint i0, i1, i2;
    fetch_hit_indices(mesh, i0, i1, i2);

    vec3 world_normal = fetch_world_normal(mesh, i0, i1, i2);
    if (gl_HitKindEXT == gl_HitKindBackFacingTriangleEXT)
        world_normal = -world_normal;

    vec2 uv = fetch_uv(mesh, i0, i1, i2);

    // ------------------------------------------------------------------ Material resolve
    vec3  base_color = mat.diffuse;
    float roughness  = mat.roughness;
    float metallic   = mat.metallic;
    vec3  specular   = mat.specular;  // pre-computed F0; overridden if mr_tex present

    if (mat.diffuse_tex >= 0)
        base_color *= texture(textures[nonuniformEXT(mat.diffuse_tex)], uv).rgb;

    if (mat.mr_tex >= 0) {
        vec4 mr  = texture(textures[nonuniformEXT(mat.mr_tex)], uv);
        roughness *= mr.g;   // G channel = roughness
        metallic  *= mr.b;   // B channel = metallic
        specular   = mix(vec3(0.04), base_color, metallic);
    }

    // ------------------------------------------------------------------ Normal map (TBN)
    if (mat.normal_tex >= 0) {
        FloatBuf verts = FloatBuf(mesh.vertex_addr);
        vec3 p0 = vec3(verts.d[i0*8u+0u], verts.d[i0*8u+1u], verts.d[i0*8u+2u]);
        vec3 p1 = vec3(verts.d[i1*8u+0u], verts.d[i1*8u+1u], verts.d[i1*8u+2u]);
        vec3 p2 = vec3(verts.d[i2*8u+0u], verts.d[i2*8u+1u], verts.d[i2*8u+2u]);
        vec2 uv0 = vec2(verts.d[i0*8u+6u], verts.d[i0*8u+7u]);
        vec2 uv1 = vec2(verts.d[i1*8u+6u], verts.d[i1*8u+7u]);
        vec2 uv2 = vec2(verts.d[i2*8u+6u], verts.d[i2*8u+7u]);

        vec3  e1   = p1 - p0;
        vec3  e2   = p2 - p0;
        vec2  duv1 = uv1 - uv0;
        vec2  duv2 = uv2 - uv0;
        float det  = duv1.x * duv2.y - duv2.x * duv1.y;

        vec3 T_world, B_world;
        if (abs(det) > 1e-6) {
            float f  = 1.0 / det;
            vec3 T_os = f * (duv2.y * e1 - duv1.y * e2);
            T_world   = normalize(mat3(gl_ObjectToWorldEXT) * T_os);
            T_world   = normalize(T_world - dot(T_world, world_normal) * world_normal);
            B_world   = cross(world_normal, T_world);
        } else {
            // Degenerate UV: fallback orthonormal frame
            vec3 up = abs(world_normal.y) < 0.999 ? vec3(0,1,0) : vec3(1,0,0);
            T_world = normalize(cross(up, world_normal));
            B_world = cross(world_normal, T_world);
        }

        vec3 nm = texture(textures[nonuniformEXT(mat.normal_tex)], uv).rgb * 2.0 - 1.0;
        world_normal = normalize(T_world * nm.x + B_world * nm.y + world_normal * nm.z);
    }

    vec3 emissive = mat.emissive;
    if (mat.emissive_tex >= 0)
        emissive *= texture(textures[nonuniformEXT(mat.emissive_tex)], uv).rgb;

    payload.hit        = 1;
    payload.position   = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    payload.normal     = world_normal;
    payload.diffuse    = base_color * (1.0 - metallic);
    payload.specular   = specular;
    payload.emissive   = emissive;
    payload.absorption = vec3(0.0);
    payload.roughness  = roughness;
    payload.ior        = 0.0;
}
