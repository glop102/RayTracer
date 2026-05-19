#version 460
#extension GL_EXT_ray_tracing                            : require
#extension GL_EXT_buffer_reference                       : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer FloatBuf { float d[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer UintBuf  { uint  d[]; };

struct MeshRef      { uint64_t vertex_addr; uint64_t index_addr; };
struct GpuMaterial  { vec3 diffuse; float roughness; vec3 specular; float ior; vec3 emissive; float _pad1; vec3 absorption; float _pad2; };
struct InstanceData { uint mesh_idx; uint mat_idx; uint _pad[2]; };

layout(set = 0, binding = 2, std430) readonly buffer MeshRefs  { MeshRef     d[]; } mesh_refs;
layout(set = 0, binding = 3, std430) readonly buffer Materials { GpuMaterial d[]; } materials;
layout(set = 0, binding = 4, std430) readonly buffer Instances { InstanceData d[]; } instances;

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

hitAttributeEXT vec2 bary;

void main() {
    uint         iid  = gl_InstanceCustomIndexEXT;
    InstanceData inst = instances.d[iid];
    MeshRef      mesh = mesh_refs.d[inst.mesh_idx];
    GpuMaterial  mat  = materials.d[inst.mat_idx];

    FloatBuf verts = FloatBuf(mesh.vertex_addr);
    UintBuf  inds  = UintBuf(mesh.index_addr);

    uint base = uint(gl_PrimitiveID) * 3u;
    uint i0 = inds.d[base];
    uint i1 = inds.d[base + 1u];
    uint i2 = inds.d[base + 2u];

    vec3 n0 = vec3(verts.d[i0*6u+3u], verts.d[i0*6u+4u], verts.d[i0*6u+5u]);
    vec3 n1 = vec3(verts.d[i1*6u+3u], verts.d[i1*6u+4u], verts.d[i1*6u+5u]);
    vec3 n2 = vec3(verts.d[i2*6u+3u], verts.d[i2*6u+4u], verts.d[i2*6u+5u]);

    float w           = 1.0 - bary.x - bary.y;
    vec3 interp_n     = normalize(w * n0 + bary.x * n1 + bary.y * n2);
    vec3 world_normal = normalize(mat3(gl_ObjectToWorldEXT) * interp_n);
    // Normal is always outward-facing (no back-face flip).
    // Raygen uses sign of payload.ior to determine entry vs exit.

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
