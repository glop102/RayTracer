#version 460
#extension GL_EXT_ray_tracing                            : require
#extension GL_EXT_buffer_reference                       : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Buffer reference types for reading mesh data via raw device addresses.
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer FloatBuf { float d[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer UintBuf  { uint  d[]; };

// Scene description SSBOs (indexed by gl_InstanceCustomIndexEXT / mesh_index).
struct MeshRef     { uint64_t vertex_addr; uint64_t index_addr; };
struct GpuMaterial { vec3 diffuse; float roughness; vec3 specular; float _pad0; vec3 emissive; float _pad1; };
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
    float roughness;
    int   hit;
};
layout(location = 0) rayPayloadInEXT HitResult payload;

hitAttributeEXT vec2 bary;

void main() {
    // Resolve instance → mesh + material.
    uint         iid  = gl_InstanceCustomIndexEXT;
    InstanceData inst = instances.d[iid];
    MeshRef      mesh = mesh_refs.d[inst.mesh_idx];
    GpuMaterial  mat  = materials.d[inst.mat_idx];

    // Fetch the triangle's vertex positions via buffer_reference.
    FloatBuf verts = FloatBuf(mesh.vertex_addr);
    UintBuf  inds  = UintBuf(mesh.index_addr);

    uint base = uint(gl_PrimitiveID) * 3u;
    uint i0 = inds.d[base];
    uint i1 = inds.d[base + 1u];
    uint i2 = inds.d[base + 2u];

    vec3 v0 = vec3(verts.d[i0*3u], verts.d[i0*3u+1u], verts.d[i0*3u+2u]);
    vec3 v1 = vec3(verts.d[i1*3u], verts.d[i1*3u+1u], verts.d[i1*3u+2u]);
    vec3 v2 = vec3(verts.d[i2*3u], verts.d[i2*3u+1u], verts.d[i2*3u+2u]);

    // Face normal in object space → world space.
    // mat3(gl_ObjectToWorldEXT) is correct for uniform-scale transforms.
    vec3 obj_normal   = normalize(cross(v1 - v0, v2 - v0));
    vec3 world_normal = normalize(mat3(gl_ObjectToWorldEXT) * obj_normal);
    if (gl_HitKindEXT == gl_HitKindBackFacingTriangleEXT)
        world_normal = -world_normal;

    payload.hit       = 1;
    payload.position  = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    payload.normal    = world_normal;
    payload.diffuse   = mat.diffuse;
    payload.specular  = mat.specular;
    payload.emissive  = mat.emissive;
    payload.roughness = mat.roughness;
}
