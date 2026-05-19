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

    vec3 world_normal = fetch_world_normal(mesh);
    if (gl_HitKindEXT == gl_HitKindBackFacingTriangleEXT)
        world_normal = -world_normal;

    payload.hit        = 1;
    payload.position   = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    payload.normal     = world_normal;
    payload.diffuse    = mat.diffuse;
    payload.specular   = mat.specular;
    payload.emissive   = mat.emissive;
    payload.absorption = vec3(0.0);
    payload.roughness  = mat.roughness;
    payload.ior        = 0.0;
}
