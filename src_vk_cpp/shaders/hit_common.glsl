// Shared declarations for all triangle closest-hit shaders.
// #include this after the #version and #extension directives.

// Vertex buffer is interleaved {x,y,z, nx,ny,nz} — stride 6 floats (24 bytes).
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer FloatBuf { float d[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer UintBuf  { uint  d[]; };

struct MeshRef      { uint64_t vertex_addr; uint64_t index_addr; };
struct GpuMaterial  { vec3 diffuse; float roughness; vec3 specular; float ior; vec3 emissive; float _pad1; vec3 absorption; float _pad2; };
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

// Fetch the barycentric-interpolated smooth normal at the current hit,
// transformed into world space. Vertex normals sit at stride offset +3 in the
// interleaved buffer; bary and gl_PrimitiveID / gl_ObjectToWorldEXT are builtins.
vec3 fetch_world_normal(MeshRef mesh) {
    FloatBuf verts = FloatBuf(mesh.vertex_addr);
    UintBuf  inds  = UintBuf(mesh.index_addr);

    uint base = uint(gl_PrimitiveID) * 3u;
    uint i0 = inds.d[base];
    uint i1 = inds.d[base + 1u];
    uint i2 = inds.d[base + 2u];

    // Fetch per-vertex smooth normals from the interleaved buffer (offset +3).
    vec3 n0 = vec3(verts.d[i0*6u+3u], verts.d[i0*6u+4u], verts.d[i0*6u+5u]);
    vec3 n1 = vec3(verts.d[i1*6u+3u], verts.d[i1*6u+4u], verts.d[i1*6u+5u]);
    vec3 n2 = vec3(verts.d[i2*6u+3u], verts.d[i2*6u+4u], verts.d[i2*6u+5u]);

    // Barycentric interpolation: bary.x = u, bary.y = v, w = 1 - u - v.
    float w = 1.0 - bary.x - bary.y;
    vec3 interp_n = normalize(w * n0 + bary.x * n1 + bary.y * n2);
    return normalize(mat3(gl_ObjectToWorldEXT) * interp_n);
}
