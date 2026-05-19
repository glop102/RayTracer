// Shared declarations for all triangle closest-hit shaders.
// #include this after the #version and #extension directives.

// Vertex buffer is interleaved {x,y,z, nx,ny,nz, u,v} — stride 8 floats (32 bytes).
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer FloatBuf { float d[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer UintBuf  { uint  d[]; };

struct MeshRef      { uint64_t vertex_addr; uint64_t index_addr; };
struct GpuMaterial  {
    vec3  diffuse;    float roughness;
    vec3  specular;   float ior;
    vec3  emissive;   float metallic;   // PBR metallic factor
    vec3  absorption; float _pad2;
    int   diffuse_tex; int mr_tex; int normal_tex; int _pad3;
};
struct InstanceData { uint mesh_idx; uint mat_idx; uint _pad[2]; };

layout(set = 0, binding = 2, std430) readonly buffer MeshRefs  { MeshRef      d[]; } mesh_refs;
layout(set = 0, binding = 3, std430) readonly buffer Materials { GpuMaterial  d[]; } materials;
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

// Fetch the three vertex indices for the current hit primitive.
void fetch_hit_indices(MeshRef mesh, out uint i0, out uint i1, out uint i2) {
    UintBuf inds = UintBuf(mesh.index_addr);
    uint base = uint(gl_PrimitiveID) * 3u;
    i0 = inds.d[base];
    i1 = inds.d[base + 1u];
    i2 = inds.d[base + 2u];
}

// Barycentric-interpolated smooth normal transformed to world space.
vec3 fetch_world_normal(MeshRef mesh, uint i0, uint i1, uint i2) {
    FloatBuf verts = FloatBuf(mesh.vertex_addr);
    vec3 n0 = vec3(verts.d[i0*8u+3u], verts.d[i0*8u+4u], verts.d[i0*8u+5u]);
    vec3 n1 = vec3(verts.d[i1*8u+3u], verts.d[i1*8u+4u], verts.d[i1*8u+5u]);
    vec3 n2 = vec3(verts.d[i2*8u+3u], verts.d[i2*8u+4u], verts.d[i2*8u+5u]);
    float w = 1.0 - bary.x - bary.y;
    return normalize(mat3(gl_ObjectToWorldEXT) * normalize(w*n0 + bary.x*n1 + bary.y*n2));
}

// Barycentric-interpolated UV.
vec2 fetch_uv(MeshRef mesh, uint i0, uint i1, uint i2) {
    FloatBuf verts = FloatBuf(mesh.vertex_addr);
    vec2 uv0 = vec2(verts.d[i0*8u+6u], verts.d[i0*8u+7u]);
    vec2 uv1 = vec2(verts.d[i1*8u+6u], verts.d[i1*8u+7u]);
    vec2 uv2 = vec2(verts.d[i2*8u+6u], verts.d[i2*8u+7u]);
    float w = 1.0 - bary.x - bary.y;
    return w*uv0 + bary.x*uv1 + bary.y*uv2;
}
