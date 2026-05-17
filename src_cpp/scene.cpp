#include "scene.h"
#include <utility>
#include <functional>
#include <cstring>
#include <immintrin.h>

using std::shared_ptr;
using std::make_shared;

bool HittableList::hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const{
    bool found_hit = false;
    for(int x=0; x<objects.size(); x++){
        found_hit |= objects[x]->hit(ray,allowed_distance,rec);
    }
    return found_hit;
}

void HittableList::add(shared_ptr<Hittable> object){
    objects.push_back(object);
}
void HittableList::clear(){
    objects.clear();
}

BBox HittableList::bbox()const{
    switch(objects.size()){
        case 0: return BBox{0};
        case 1: return objects[0]->bbox();
        default:
            BBox b = objects[0]->bbox();
            for(int x=1; x<objects.size(); x++){
                BBox ob = objects[x]->bbox();
                Vector3::min_accum(b.min,ob.min);
                Vector3::max_accum(b.max,ob.max);
            }
            return b;
    }
}


//===================================================================
// Instance
//===================================================================

Instance::Instance(ObjList& objects, int max_depth)
    : translation{0,0,0} {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R[i][j] = (i == j) ? 1.0 : 0.0;

    bool all_triangles = !objects.empty();
    for (auto& obj : objects)
        if (!dynamic_cast<Triangle*>(obj.get())) { all_triangles = false; break; }

    if (all_triangles) {
        std::vector<Triangle> tris;
        tris.reserve(objects.size());
        for (auto& obj : objects)
            tris.push_back(*static_cast<Triangle*>(obj.get()));
        bvh = std::make_shared<TriangleBVH8>(tris, max_depth);
    } else {
        bvh = std::make_shared<BVH8List>(objects, max_depth);
    }
    recompute_bbox();
}

Vector3 Instance::rot(const Vector3& v) const {
    return {
        R[0][0]*v.x + R[0][1]*v.y + R[0][2]*v.z,
        R[1][0]*v.x + R[1][1]*v.y + R[1][2]*v.z,
        R[2][0]*v.x + R[2][1]*v.y + R[2][2]*v.z,
    };
}
Vector3 Instance::rot_inv(const Vector3& v) const {
    // R is orthogonal so R^-1 = R^T
    return {
        R[0][0]*v.x + R[1][0]*v.y + R[2][0]*v.z,
        R[0][1]*v.x + R[1][1]*v.y + R[2][1]*v.z,
        R[0][2]*v.x + R[1][2]*v.y + R[2][2]*v.z,
    };
}

void Instance::recompute_bbox() {
    BBox b = bvh->bbox();
    memoized_bbox = BBox{};
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                Point3 corner{
                    i ? b.max.x : b.min.x,
                    j ? b.max.y : b.min.y,
                    k ? b.max.z : b.min.z,
                };
                memoized_bbox.absorb(rot(corner) + translation);
            }
}

Instance& Instance::rotate_y(double degrees) {
    double rad = degrees * PI / 180.0;
    double c = std::cos(rad), s = std::sin(rad);
    // Y rotation matrix: x'=cx+sz, y'=y, z'=-sx+cz
    double Ry[3][3] = {{c,0,s},{0,1,0},{-s,0,c}};
    // Compose: R = Ry * R
    double nr[3][3] = {};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                nr[i][j] += Ry[i][k] * R[k][j];
    std::memcpy(R, nr, sizeof(R));
    recompute_bbox();
    return *this;
}

Instance& Instance::translate(const Vector3& offset) {
    translation += offset;
    recompute_bbox();
    return *this;
}

bool Instance::hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec) const {
    // Transform ray to object space: R^T * (origin - t), R^T * direction
    Ray obj_ray(
        rot_inv(ray.origin - translation),
        rot_inv(ray.direction)
    );
    if (!bvh->hit(obj_ray, allowed_distance, rec)) return false;
    // Transform result back to world space
    rec.intersection_point = rot(rec.intersection_point) + translation;
    rec.normal = rot(rec.normal);
    return true;
}

BBox Instance::bbox() const { return memoized_bbox; }

void Instance::debug_print_tree(int indent) const {
    bvh->debug_print_tree(indent);
}




//===================================================================
// BVH8List
//===================================================================

BVH8List::BVH8List(ObjList& world_objects, int max_depth) {
    if (world_objects.empty()) return;

    root_bbox = world_objects[0]->bbox();
    for (size_t i = 1; i < world_objects.size(); i++)
        root_bbox.absorb(world_objects[i]->bbox());

    // Reserve generously to avoid reallocations invalidating pointers in the tree
    nodes.reserve(world_objects.size() / 2 + 8);
    nodes.emplace_back(); // root is nodes[0]
    build(0, world_objects, max_depth);
}

std::pair<ObjList,ObjList> BVH8List::sah_split(ObjList& objs) {
    if (objs.size() <= 1) return {objs, {}};

    BBox centroid_bbox;
    for (auto& obj : objs)
        centroid_bbox.absorb(obj->bbox().center());

    auto delta = centroid_bbox.max - centroid_bbox.min;

    int axis = 0;
    if (delta.y > delta.x && delta.y > delta.z) axis = 1;
    else if (delta.z > delta.x) axis = 2;

    double range = delta[axis];
    double axis_min = centroid_bbox.min[axis];

    if (range < 1e-10) {
        // All centroids coincide - just split half/half
        size_t half = objs.size() / 2;
        return { ObjList(objs.begin(), objs.begin()+half),
                 ObjList(objs.begin()+half, objs.end()) };
    }

    const int NUM_BINS = std::max(8, (int)sqrt((double)objs.size()));
    struct Bin { ObjList objects; BBox bbox; };
    std::vector<Bin> bins(NUM_BINS);

    for (auto& obj : objs) {
        double coord = obj->bbox().center()[axis];
        int b = (int)((coord - axis_min) / range * NUM_BINS * 0.99999);
        bins[b].bbox.absorb(obj->bbox());
        bins[b].objects.push_back(obj);
    }

    int best_split = 1;
    double best_cost = std::numeric_limits<double>::max();
    for (int s = 1; s < NUM_BINS; s++) {
        BBox lb, rb;
        int lc = 0, rc = 0;
        for (int i = 0;   i < s;        i++) { if (bins[i].objects.empty()) continue; lb.absorb(bins[i].bbox); lc += bins[i].objects.size(); }
        for (int i = s;   i < NUM_BINS;  i++) { if (bins[i].objects.empty()) continue; rb.absorb(bins[i].bbox); rc += bins[i].objects.size(); }
        if (!lc || !rc) continue;
        double cost = lc * lb.half_surface_area() + rc * rb.half_surface_area();
        if (cost < best_cost) { best_cost = cost; best_split = s; }
    }

    ObjList left_objs, right_objs;
    for (int i = 0;          i < best_split; i++) for (auto& o : bins[i].objects) left_objs.push_back(o);
    for (int i = best_split; i < NUM_BINS;   i++) for (auto& o : bins[i].objects) right_objs.push_back(o);

    if (left_objs.empty() || right_objs.empty()) {
        size_t half = objs.size() / 2;
        return { ObjList(objs.begin(), objs.begin()+half),
                 ObjList(objs.begin()+half, objs.end()) };
    }
    return {left_objs, right_objs};
}

void BVH8List::build(int node_idx, ObjList objs, int depth) {
    // Greedy 7-split: repeatedly split the largest group until we have up to 8 groups
    std::vector<ObjList> groups;
    groups.reserve(8);
    groups.push_back(std::move(objs));

    while ((int)groups.size() < 8) {
        size_t best_i = 0;
        for (size_t i = 1; i < groups.size(); i++)
            if (groups[i].size() > groups[best_i].size()) best_i = i;
        if (groups[best_i].size() <= 1) break;

        auto [left, right] = sah_split(groups[best_i]);
        if (left.empty() || right.empty()) break;
        groups.erase(groups.begin() + best_i);
        groups.push_back(std::move(left));
        groups.push_back(std::move(right));
    }

    nodes[node_idx].num_children = (uint8_t)groups.size();

    for (int i = 0; i < (int)groups.size(); i++) {
        BBox gb;
        for (auto& obj : groups[i]) gb.absorb(obj->bbox());

        nodes[node_idx].min_x[i] = (float)gb.min.x;
        nodes[node_idx].min_y[i] = (float)gb.min.y;
        nodes[node_idx].min_z[i] = (float)gb.min.z;
        nodes[node_idx].max_x[i] = (float)gb.max.x;
        nodes[node_idx].max_y[i] = (float)gb.max.y;
        nodes[node_idx].max_z[i] = (float)gb.max.z;

        if (depth <= 1 || groups[i].size() <= 4) {
            int leaf_idx = (int)leaves.size();
            Leaf leaf;
            leaf.obj_start = (int)flat_objects.size();
            leaf.obj_count = (int)groups[i].size();
            leaves.push_back(leaf);
            for (auto& obj : groups[i]) flat_objects.push_back(obj);
            nodes[node_idx].child[i] = ~leaf_idx;
        } else {
            int child_node_idx = (int)nodes.size();
            nodes[node_idx].child[i] = child_node_idx;
            nodes.emplace_back(); // may reallocate - always access via index after this
            build(child_node_idx, std::move(groups[i]), depth - 1);
        }
    }
}

__attribute__((target("avx2,fma")))
static int aabb_test_8(const BVH8Node& node,
                           float ox, float oy, float oz,
                           float idx, float idy, float idz,
                           float t_near, float t_far,
                           float* tmin_out)
{
    __m256 vox  = _mm256_set1_ps(ox);
    __m256 voy  = _mm256_set1_ps(oy);
    __m256 voz  = _mm256_set1_ps(oz);
    __m256 vidx = _mm256_set1_ps(idx);
    __m256 vidy = _mm256_set1_ps(idy);
    __m256 vidz = _mm256_set1_ps(idz);
    __m256 vtnear = _mm256_set1_ps(t_near);
    __m256 vtfar  = _mm256_set1_ps(t_far);

    __m256 min_x = _mm256_load_ps(node.min_x);
    __m256 max_x = _mm256_load_ps(node.max_x);
    __m256 min_y = _mm256_load_ps(node.min_y);
    __m256 max_y = _mm256_load_ps(node.max_y);
    __m256 min_z = _mm256_load_ps(node.min_z);
    __m256 max_z = _mm256_load_ps(node.max_z);

    // t values for each slab
    __m256 tx0 = _mm256_mul_ps(_mm256_sub_ps(min_x, vox), vidx);
    __m256 tx1 = _mm256_mul_ps(_mm256_sub_ps(max_x, vox), vidx);
    __m256 ty0 = _mm256_mul_ps(_mm256_sub_ps(min_y, voy), vidy);
    __m256 ty1 = _mm256_mul_ps(_mm256_sub_ps(max_y, voy), vidy);
    __m256 tz0 = _mm256_mul_ps(_mm256_sub_ps(min_z, voz), vidz);
    __m256 tz1 = _mm256_mul_ps(_mm256_sub_ps(max_z, voz), vidz);

    __m256 tmin = _mm256_max_ps(
        _mm256_max_ps(_mm256_min_ps(tx0, tx1), _mm256_min_ps(ty0, ty1)),
        _mm256_min_ps(tz0, tz1));
    __m256 tmax = _mm256_min_ps(
        _mm256_min_ps(_mm256_max_ps(tx0, tx1), _mm256_max_ps(ty0, ty1)),
        _mm256_max_ps(tz0, tz1));

    __m256 hit = _mm256_and_ps(
        _mm256_and_ps(
            _mm256_cmp_ps(tmax, tmin,   _CMP_GE_OQ),  // tmax >= tmin
            _mm256_cmp_ps(tmax, vtnear, _CMP_GT_OQ)), // tmax > t_near
            _mm256_cmp_ps(tmin, vtfar,  _CMP_LT_OQ)); // tmin < t_far

    _mm256_storeu_ps(tmin_out, tmin);
    return _mm256_movemask_ps(hit);
}

bool BVH8List::hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec) const {
    if (nodes.empty()) return false;

    float ox  = (float)ray.origin.x,        oy  = (float)ray.origin.y,        oz  = (float)ray.origin.z;
    float idx = (float)ray.inv_direction.x,  idy = (float)ray.inv_direction.y,  idz = (float)ray.inv_direction.z;

    struct StackEntry { int node_idx; float t_near; };
    thread_local std::vector<StackEntry> stack;
    stack.clear();
    stack.push_back({0, (float)allowed_distance.min});

    bool found_hit = false;
    alignas(32) float tmin_out[8];

    while (!stack.empty()) {
        auto [ni, t_near_entry] = stack.back();
        stack.pop_back();

        float t_far = (float)allowed_distance.max;
        if (t_near_entry >= t_far) continue;

        const BVH8Node& node = nodes[ni];
        int mask = aabb_test_8(node, ox, oy, oz, idx, idy, idz,
                               (float)allowed_distance.min, t_far, tmin_out);
        mask &= (1 << node.num_children) - 1;
        if (!mask) continue;

        // Collect hits, sort by tmin descending so closest is processed first (LIFO stack)
        struct Hit { int child; float tmin; };
        Hit hits[8];
        int nh = 0;
        for (int m = mask; m; m &= m-1) {
            int i = __builtin_ctz(m);
            hits[nh++] = {node.child[i], tmin_out[i]};
        }
        // Simple insertion sort (up to 8 items)
        for (int a = 1; a < nh; a++) {
            Hit key = hits[a];
            int b = a - 1;
            while (b >= 0 && hits[b].tmin < key.tmin) { hits[b+1] = hits[b]; b--; }
            hits[b+1] = key;
        }

        for (int h = 0; h < nh; h++) {
            int child = hits[h].child;
            if (child < 0) {
                // Leaf - test all objects
                const Leaf& leaf = leaves[~child];
                for (int j = leaf.obj_start; j < leaf.obj_start + leaf.obj_count; j++)
                    found_hit |= flat_objects[j]->hit(ray, allowed_distance, rec);
            } else {
                stack.push_back({child, hits[h].tmin});
            }
        }
    }
    return found_hit;
}

BBox BVH8List::bbox() const {
    return root_bbox;
}

void BVH8List::debug_print_tree(int indent) const {
    size_t total_leaf_objs = 0;
    size_t max_leaf_size = 0;
    for (auto& leaf : leaves) {
        total_leaf_objs += leaf.obj_count;
        if ((size_t)leaf.obj_count > max_leaf_size) max_leaf_size = leaf.obj_count;
    }

    // Measure tree depth via BFS
    int max_depth = 0;
    std::vector<std::pair<int,int>> bfs_stack; // {node_idx, depth}
    bfs_stack.push_back({0, 0});
    while (!bfs_stack.empty()) {
        auto [ni, d] = bfs_stack.back();
        bfs_stack.pop_back();
        if (d > max_depth) max_depth = d;
        const BVH8Node& node = nodes[ni];
        for (int i = 0; i < node.num_children; i++) {
            if (node.child[i] >= 0)
                bfs_stack.push_back({node.child[i], d+1});
        }
    }

    double avg_leaf = leaves.empty() ? 0.0 : (double)total_leaf_objs / leaves.size();
    printf("BVH8List: %zu nodes, %zu leaves, %zu objects\n",
           nodes.size(), leaves.size(), flat_objects.size());
    printf("  Max depth: %d  |  Avg leaf size: %.1f  |  Max leaf size: %zu\n",
           max_depth, avg_leaf, max_leaf_size);

    for (auto& obj : flat_objects)
        obj->debug_print_tree(indent + 1);
}


//===================================================================
// TriangleBVH8
//===================================================================

TriangleBVH8::TriangleBVH8(std::vector<Triangle>& world_tris, int max_depth) {
    if (world_tris.empty()) return;

    for (auto& tri : world_tris) {
        bool found = false;
        for (auto& m : material_storage)
            if (m.get() == tri.material.get()) { found = true; break; }
        if (!found) material_storage.push_back(tri.material);
    }

    root_bbox = world_tris[0].bbox();
    for (size_t i = 1; i < world_tris.size(); i++)
        root_bbox.absorb(world_tris[i].bbox());

    nodes.reserve(world_tris.size() / 4 + 8);
    nodes.emplace_back();
    build(0, world_tris, max_depth);
}

std::pair<std::vector<Triangle>, std::vector<Triangle>>
TriangleBVH8::sah_split(std::vector<Triangle>& tris) {
    if (tris.size() <= 1) return {tris, {}};

    BBox centroid_bbox;
    for (auto& tri : tris)
        centroid_bbox.absorb(tri.bbox().center());

    auto delta = centroid_bbox.max - centroid_bbox.min;
    int axis = 0;
    if (delta.y > delta.x && delta.y > delta.z) axis = 1;
    else if (delta.z > delta.x) axis = 2;

    double range = delta[axis];
    double axis_min = centroid_bbox.min[axis];

    if (range < 1e-10) {
        size_t half = tris.size() / 2;
        return { std::vector<Triangle>(tris.begin(), tris.begin()+half),
                 std::vector<Triangle>(tris.begin()+half, tris.end()) };
    }

    const int NUM_BINS = std::max(8, (int)sqrt((double)tris.size()));
    struct Bin { std::vector<Triangle> tris; BBox bbox; };
    std::vector<Bin> bins(NUM_BINS);

    for (auto& tri : tris) {
        BBox tb = tri.bbox();
        double coord = tb.center()[axis];
        int b = (int)((coord - axis_min) / range * NUM_BINS * 0.99999);
        bins[b].bbox.absorb(tb);
        bins[b].tris.push_back(tri);
    }

    int best_split = 1;
    double best_cost = std::numeric_limits<double>::max();
    for (int s = 1; s < NUM_BINS; s++) {
        BBox lb, rb;
        int lc = 0, rc = 0;
        for (int i = 0; i < s;        i++) { if (bins[i].tris.empty()) continue; lb.absorb(bins[i].bbox); lc += bins[i].tris.size(); }
        for (int i = s; i < NUM_BINS;  i++) { if (bins[i].tris.empty()) continue; rb.absorb(bins[i].bbox); rc += bins[i].tris.size(); }
        if (!lc || !rc) continue;
        double cost = lc * lb.half_surface_area() + rc * rb.half_surface_area();
        if (cost < best_cost) { best_cost = cost; best_split = s; }
    }

    std::vector<Triangle> left_tris, right_tris;
    for (int i = 0;          i < best_split; i++) for (auto& t : bins[i].tris) left_tris.push_back(std::move(t));
    for (int i = best_split; i < NUM_BINS;   i++) for (auto& t : bins[i].tris) right_tris.push_back(std::move(t));

    if (left_tris.empty() || right_tris.empty()) {
        size_t half = tris.size() / 2;
        return { std::vector<Triangle>(tris.begin(), tris.begin()+half),
                 std::vector<Triangle>(tris.begin()+half, tris.end()) };
    }
    return {std::move(left_tris), std::move(right_tris)};
}

void TriangleBVH8::build(int node_idx, std::vector<Triangle> tris, int depth) {
    std::vector<std::vector<Triangle>> groups;
    groups.reserve(8);
    groups.push_back(std::move(tris));

    while ((int)groups.size() < 8) {
        size_t best_i = 0;
        for (size_t i = 1; i < groups.size(); i++)
            if (groups[i].size() > groups[best_i].size()) best_i = i;
        if (groups[best_i].size() <= 8) break;  // stop when all groups fit in one SoA leaf

        auto [left, right] = sah_split(groups[best_i]);
        if (left.empty() || right.empty()) break;
        groups.erase(groups.begin() + best_i);
        groups.push_back(std::move(left));
        groups.push_back(std::move(right));
    }

    nodes[node_idx].num_children = (uint8_t)groups.size();

    for (int i = 0; i < (int)groups.size(); i++) {
        BBox gb;
        for (auto& tri : groups[i]) gb.absorb(tri.bbox());

        nodes[node_idx].min_x[i] = (float)gb.min.x;
        nodes[node_idx].min_y[i] = (float)gb.min.y;
        nodes[node_idx].min_z[i] = (float)gb.min.z;
        nodes[node_idx].max_x[i] = (float)gb.max.x;
        nodes[node_idx].max_y[i] = (float)gb.max.y;
        nodes[node_idx].max_z[i] = (float)gb.max.z;

        if (depth <= 1 || groups[i].size() <= 8) {
            SoALeaf leaf{};
            leaf.count = (uint8_t)groups[i].size();
            for (int j = 0; j < (int)groups[i].size(); j++) {
                const Triangle& tri = groups[i][j];
                leaf.p1_x[j] = (float)tri.p1.x;  leaf.p1_y[j] = (float)tri.p1.y;  leaf.p1_z[j] = (float)tri.p1.z;
                leaf.e1_x[j] = (float)tri.e1.x;  leaf.e1_y[j] = (float)tri.e1.y;  leaf.e1_z[j] = (float)tri.e1.z;
                leaf.e2_x[j] = (float)tri.e2.x;  leaf.e2_y[j] = (float)tri.e2.y;  leaf.e2_z[j] = (float)tri.e2.z;
                leaf.nx[j]   = (float)tri.normal.x; leaf.ny[j] = (float)tri.normal.y; leaf.nz[j] = (float)tri.normal.z;
                leaf.mat[j]  = tri.material.get();
            }
            // Pad remaining lanes: e1=e2=0 → a=0 → always misses the epsilon check
            for (int j = (int)groups[i].size(); j < 8; j++) {
                leaf.p1_x[j] = leaf.p1_y[j] = leaf.p1_z[j] = 0.0f;
                leaf.e1_x[j] = leaf.e1_y[j] = leaf.e1_z[j] = 0.0f;
                leaf.e2_x[j] = leaf.e2_y[j] = leaf.e2_z[j] = 0.0f;
                leaf.nx[j]   = leaf.ny[j]   = leaf.nz[j]   = 0.0f;
                leaf.mat[j]  = nullptr;
            }
            nodes[node_idx].child[i] = ~(int)soa_leaves.size();
            soa_leaves.push_back(leaf);
        } else {
            int child_node_idx = (int)nodes.size();
            nodes[node_idx].child[i] = child_node_idx;
            nodes.emplace_back();
            build(child_node_idx, std::move(groups[i]), depth - 1);
        }
    }
}

__attribute__((target("avx2,fma")))
int TriangleBVH8::moller_trumbore_8(const SoALeaf& leaf,
                                     float ox, float oy, float oz,
                                     float dx, float dy, float dz,
                                     float tmin, float tmax,
                                     float* t_out)
{
    const __m256 vox = _mm256_set1_ps(ox), voy = _mm256_set1_ps(oy), voz = _mm256_set1_ps(oz);
    const __m256 vdx = _mm256_set1_ps(dx), vdy = _mm256_set1_ps(dy), vdz = _mm256_set1_ps(dz);

    const __m256 e1x = _mm256_load_ps(leaf.e1_x), e1y = _mm256_load_ps(leaf.e1_y), e1z = _mm256_load_ps(leaf.e1_z);
    const __m256 e2x = _mm256_load_ps(leaf.e2_x), e2y = _mm256_load_ps(leaf.e2_y), e2z = _mm256_load_ps(leaf.e2_z);
    const __m256 p1x = _mm256_load_ps(leaf.p1_x), p1y = _mm256_load_ps(leaf.p1_y), p1z = _mm256_load_ps(leaf.p1_z);

    // h = dir × e2
    const __m256 hx = _mm256_fmsub_ps(vdy, e2z, _mm256_mul_ps(vdz, e2y));
    const __m256 hy = _mm256_fmsub_ps(vdz, e2x, _mm256_mul_ps(vdx, e2z));
    const __m256 hz = _mm256_fmsub_ps(vdx, e2y, _mm256_mul_ps(vdy, e2x));

    // a = e1 · h; reject if |a| < epsilon (ray parallel to triangle)
    const __m256 a = _mm256_fmadd_ps(e1x, hx, _mm256_fmadd_ps(e1y, hy, _mm256_mul_ps(e1z, hz)));
    const __m256 abs_a = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a);
    __m256 active = _mm256_cmp_ps(abs_a, _mm256_set1_ps(1e-8f), _CMP_GE_OQ);

    // f = 1/a
    const __m256 f = _mm256_div_ps(_mm256_set1_ps(1.0f), a);

    // s = origin - p1
    const __m256 sx = _mm256_sub_ps(vox, p1x);
    const __m256 sy = _mm256_sub_ps(voy, p1y);
    const __m256 sz = _mm256_sub_ps(voz, p1z);

    // u = f * (s · h); reject if u < 0 or u > 1
    const __m256 u = _mm256_mul_ps(f, _mm256_fmadd_ps(sx, hx, _mm256_fmadd_ps(sy, hy, _mm256_mul_ps(sz, hz))));
    const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    active = _mm256_and_ps(active, _mm256_cmp_ps(u, zero, _CMP_GE_OQ));
    active = _mm256_and_ps(active, _mm256_cmp_ps(u, one,  _CMP_LE_OQ));

    // q = s × e1
    const __m256 qx = _mm256_fmsub_ps(sy, e1z, _mm256_mul_ps(sz, e1y));
    const __m256 qy = _mm256_fmsub_ps(sz, e1x, _mm256_mul_ps(sx, e1z));
    const __m256 qz = _mm256_fmsub_ps(sx, e1y, _mm256_mul_ps(sy, e1x));

    // v = f * (dir · q); reject if v < 0 or u+v > 1
    const __m256 v = _mm256_mul_ps(f, _mm256_fmadd_ps(vdx, qx, _mm256_fmadd_ps(vdy, qy, _mm256_mul_ps(vdz, qz))));
    active = _mm256_and_ps(active, _mm256_cmp_ps(v, zero, _CMP_GE_OQ));
    active = _mm256_and_ps(active, _mm256_cmp_ps(_mm256_add_ps(u, v), one, _CMP_LE_OQ));

    // t = f * (e2 · q); reject if outside [tmin, tmax]
    const __m256 t = _mm256_mul_ps(f, _mm256_fmadd_ps(e2x, qx, _mm256_fmadd_ps(e2y, qy, _mm256_mul_ps(e2z, qz))));
    active = _mm256_and_ps(active, _mm256_cmp_ps(t, _mm256_set1_ps(tmin), _CMP_GT_OQ));
    active = _mm256_and_ps(active, _mm256_cmp_ps(t, _mm256_set1_ps(tmax), _CMP_LT_OQ));

    _mm256_storeu_ps(t_out, t);
    return _mm256_movemask_ps(active);
}

bool TriangleBVH8::hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec) const {
    if (nodes.empty()) return false;

    const float ox  = (float)ray.origin.x,       oy  = (float)ray.origin.y,       oz  = (float)ray.origin.z;
    const float dx  = (float)ray.direction.x,     dy  = (float)ray.direction.y,     dz  = (float)ray.direction.z;
    const float idx = (float)ray.inv_direction.x, idy = (float)ray.inv_direction.y, idz = (float)ray.inv_direction.z;

    struct StackEntry { int node_idx; float t_near; };
    thread_local std::vector<StackEntry> stack;
    stack.clear();
    stack.push_back({0, (float)allowed_distance.min});

    bool found_hit = false;
    alignas(32) float tmin_out[8];
    alignas(32) float t_out[8];

    while (!stack.empty()) {
        auto [ni, t_near_entry] = stack.back();
        stack.pop_back();

        float t_far = (float)allowed_distance.max;
        if (t_near_entry >= t_far) continue;

        const BVH8Node& node = nodes[ni];
        int mask = aabb_test_8(node, ox, oy, oz, idx, idy, idz,
                               (float)allowed_distance.min, t_far, tmin_out);
        mask &= (1 << node.num_children) - 1;
        if (!mask) continue;

        struct Hit { int child; float tmin; };
        Hit hits[8];
        int nh = 0;
        for (int m = mask; m; m &= m-1) {
            int i = __builtin_ctz(m);
            hits[nh++] = {node.child[i], tmin_out[i]};
        }
        for (int a = 1; a < nh; a++) {
            Hit key = hits[a];
            int b = a - 1;
            while (b >= 0 && hits[b].tmin < key.tmin) { hits[b+1] = hits[b]; b--; }
            hits[b+1] = key;
        }

        for (int h = 0; h < nh; h++) {
            int child = hits[h].child;
            if (child < 0) {
                const SoALeaf& leaf = soa_leaves[~child];
                int lmask = moller_trumbore_8(leaf, ox, oy, oz, dx, dy, dz,
                                              (float)allowed_distance.min,
                                              (float)allowed_distance.max, t_out);
                lmask &= (1 << leaf.count) - 1;
                for (int m = lmask; m; m &= m-1) {
                    int j = __builtin_ctz(m);
                    double t = (double)t_out[j];
                    if (t < allowed_distance.max) {
                        allowed_distance.max = t;
                        rec.distanceScale     = t;
                        rec.intersection_point = ray.at(t);
                        rec.material           = leaf.mat[j];
                        // dot(dir, stored_normal) < 0 → front face (same sign convention as scalar path)
                        if (dx*leaf.nx[j] + dy*leaf.ny[j] + dz*leaf.nz[j] < 0.0f) {
                            rec.normal     = {leaf.nx[j], leaf.ny[j], leaf.nz[j]};
                            rec.front_face = true;
                        } else {
                            rec.normal     = {-leaf.nx[j], -leaf.ny[j], -leaf.nz[j]};
                            rec.front_face = false;
                        }
                        found_hit = true;
                    }
                }
            } else {
                stack.push_back({child, hits[h].tmin});
            }
        }
    }
    return found_hit;
}

BBox TriangleBVH8::bbox() const {
    return root_bbox;
}

void TriangleBVH8::debug_print_tree(int) const {
    size_t total_tris = 0;
    size_t max_leaf_size = 0;
    for (auto& leaf : soa_leaves) {
        total_tris += leaf.count;
        if (leaf.count > max_leaf_size) max_leaf_size = leaf.count;
    }

    int max_depth = 0;
    std::vector<std::pair<int,int>> bfs_stack;
    bfs_stack.push_back({0, 0});
    while (!bfs_stack.empty()) {
        auto [ni, d] = bfs_stack.back();
        bfs_stack.pop_back();
        if (d > max_depth) max_depth = d;
        const BVH8Node& node = nodes[ni];
        for (int i = 0; i < node.num_children; i++)
            if (node.child[i] >= 0)
                bfs_stack.push_back({node.child[i], d+1});
    }

    double avg_leaf = soa_leaves.empty() ? 0.0 : (double)total_tris / soa_leaves.size();
    printf("TriangleBVH8: %zu nodes, %zu leaves, %zu triangles\n",
           nodes.size(), soa_leaves.size(), total_tris);
    printf("  Max depth: %d  |  Avg leaf size: %.1f  |  Max leaf size: %zu\n",
           max_depth, avg_leaf, max_leaf_size);
}