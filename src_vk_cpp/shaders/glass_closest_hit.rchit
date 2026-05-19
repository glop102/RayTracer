#version 460
#extension GL_EXT_ray_tracing                            : require
#extension GL_EXT_buffer_reference                       : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "hit_common.glsl"

void main() {
    uint         iid  = gl_InstanceCustomIndexEXT;
    InstanceData inst = instances.d[iid];
    MeshRef      mesh = mesh_refs.d[inst.mesh_idx];
    GpuMaterial  mat  = materials.d[inst.mat_idx];

    uint i0, i1, i2;
    fetch_hit_indices(mesh, i0, i1, i2);

    // Normal is always outward-facing (no back-face flip).
    // Raygen uses sign of payload.ior to determine entry vs exit.
    vec3 world_normal = fetch_world_normal(mesh, i0, i1, i2);

    payload.hit        = 1;
    payload.position   = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    payload.normal     = world_normal;
    payload.diffuse    = mat.diffuse;
    payload.specular   = mat.specular;
    payload.emissive   = mat.emissive;
    payload.absorption = mat.absorption;
    payload.roughness  = mat.roughness;
    // Positive ior = entering glass; negative = exiting glass.
    payload.ior = (gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT) ? mat.ior : -mat.ior;
}
