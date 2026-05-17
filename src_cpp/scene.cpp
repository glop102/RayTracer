#include "scene.h"
#include <utility>
#include <functional>
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
// BVHList
//===================================================================
BVHList::BVHList(ObjList& world_objects,int max_depth)
: max_depth_allowed(max_depth), objects(world_objects) {
    if(max_depth_allowed<=0 || objects.size() <= 2){
        // Recursion end case of max depth or only a single object
        // this should be the only case where left or right are null
        right = left = nullptr;
        if(objects.size()>=1){
            memoized_bbox = objects[0]->bbox();
            for(int x=1;x<objects.size();x++){
                memoized_bbox.absorb(objects[x]->bbox());
            }
            if(objects.size()>4) {
                printf("BVH: Warning: Leaf made with %lu objects\n    consider increasing max depth\n", objects.size());
            }
        } else {
            // Make sure the bounding box is zero size to try to never have it get hit when we have no objects to hold
            memoized_bbox.max = memoized_bbox.min = {0.0,0.0,0.0};
        }
    } else {
        memoized_bbox = objects[0]->bbox();
        BBox centroid_bbox{objects[0]->bbox().center(), objects[0]->bbox().center()};
        for(int i=1; i<objects.size(); i++){
            auto obj_bbox = objects[i]->bbox();
            memoized_bbox.absorb(obj_bbox);
            centroid_bbox.absorb(obj_bbox.center());
        }

        auto delta_centroids = centroid_bbox.max - centroid_bbox.min;
        struct BINS {
            ObjList objects;
            BBox bbox;
        };
        const int NUM_BINS = std::max(8,(int) sqrt(objects.size()));
        struct BINS bins[NUM_BINS];
        if(delta_centroids.x > delta_centroids.y && delta_centroids.x > delta_centroids.z) {
            // Longest axis is X - so lets subdivide the bounding box on that axis
            {
                auto face = objects[0];
                auto face_bbox = face->bbox();
                auto center = face_bbox.center();
                double percentage_through_centroid_range = (center.x-centroid_bbox.min.x)/delta_centroids.x;
                int bin_num = percentage_through_centroid_range*NUM_BINS*.99999;

                bins[bin_num].bbox = face_bbox;
                bins[bin_num].objects.push_back(face);
            }

            for (int x=1; x<objects.size(); x++) {
                auto face = objects[x];
                auto face_bbox = face->bbox();
                auto center = face_bbox.center();
                double percentage_through_centroid_range = (center.x-centroid_bbox.min.x)/delta_centroids.x;
                // Intentionally truncating to the floor of the double
                // If we are 20% through the x axis, then really we want 20% of NUM_BINS to put the shape into
                // The tiny fudge factor is to deal with things sometimes coming out to be exactly NUM_BINS
                int bin_num = percentage_through_centroid_range*NUM_BINS*.99999;
                bins[bin_num].bbox.absorb(face_bbox);
                bins[bin_num].objects.push_back(face);
            }
        } else if (delta_centroids.y > delta_centroids.z) {
            // Longest axis is Y
            {
                auto face = objects[0];
                auto face_bbox = face->bbox();
                auto center = face_bbox.center();
                double percentage_through_centroid_range = (center.y-centroid_bbox.min.y)/delta_centroids.y;
                int bin_num = percentage_through_centroid_range*NUM_BINS*.99999;

                bins[bin_num].bbox = face_bbox;
                bins[bin_num].objects.push_back(face);
            }

            for (int x=1; x<objects.size(); x++) {
                auto face = objects[x];
                auto face_bbox = face->bbox();
                auto center = face_bbox.center();
                double percentage_through_centroid_range = (center.y-centroid_bbox.min.y)/delta_centroids.y;
                int bin_num = percentage_through_centroid_range*NUM_BINS*.99999;
                bins[bin_num].bbox.absorb(face_bbox);
                bins[bin_num].objects.push_back(face);
            }
        } else if (delta_centroids.z > 0) {
            // Longest axis is Z
            {
                auto face = objects[0];
                auto face_bbox = face->bbox();
                auto center = face_bbox.center();
                double percentage_through_centroid_range = (center.z-centroid_bbox.min.z)/delta_centroids.z;
                int bin_num = percentage_through_centroid_range*NUM_BINS*.99999;

                bins[bin_num].bbox = face_bbox;
                bins[bin_num].objects.push_back(face);
            }

            for (int x=1; x<objects.size(); x++) {
                auto face = objects[x];
                auto face_bbox = face->bbox();
                auto center = face_bbox.center();
                double percentage_through_centroid_range = (center.z-centroid_bbox.min.z)/delta_centroids.z;
                int bin_num = percentage_through_centroid_range*NUM_BINS*.99999;
                bins[bin_num].bbox.absorb(face_bbox);
                bins[bin_num].objects.push_back(face);
            }
        } else {
            // We have no difference in center of objects and so the math explodes if we try to subdivide.
            // It is super rare to have a ton of objects with perfectly overlapping centers so lets just leave everything as is
            if ( objects.size() > 16 ) {
                printf("Warning: Numerous items overlapping with the same center and cannot be subdivided - X%f Y%f Z%f\n",
                    centroid_bbox.min.x,
                    centroid_bbox.min.y,
                    centroid_bbox.min.z
                );
            }
            left = right = nullptr;
            return;
        }

        // Now that the objects have been added into bins, lets see which bin split is the best split
        // We will Add the bounding boxes of the different bins 
        int best_bin_split = 0;
        double best_split_cost = std::numeric_limits<double>::max();
        #define OBJ_COUNT_SCALE 1.0
        for( int split=1; split<NUM_BINS; split++ ) {
            BBox temp_left_bbox,temp_right_bbox;
            int temp_left_count, temp_right_count;
            temp_left_bbox = bins[0].bbox;
            temp_left_count = bins[0].objects.size();
            temp_right_bbox = bins[NUM_BINS-1].bbox;
            temp_right_count = bins[NUM_BINS-1].objects.size();
            // Grow the left BBox
            for(int lextra=1; lextra<split; lextra++){
                if(!bins[lextra].objects.size())
                    // skip absorbing bounding boxes that have no objects in them
                    continue;
                temp_left_bbox.absorb(bins[lextra].bbox);
                temp_left_count += bins[lextra].objects.size();
            }
            //Grow the right BBox
            for(int rextra=split; rextra<NUM_BINS-1; rextra++){
                if(!bins[rextra].objects.size())
                    continue;
                temp_right_bbox.absorb(bins[rextra].bbox);
                temp_right_count += bins[rextra].objects.size();
            }
            double cost = (temp_left_count/OBJ_COUNT_SCALE)*temp_left_bbox.half_surface_area() + (temp_right_count/OBJ_COUNT_SCALE)*temp_right_bbox.half_surface_area();
            if (cost < best_split_cost){
                best_bin_split = split;
                best_split_cost = cost;
            }
        }

        // Check if we have something that doesn't make sense to break up
        if (best_split_cost >= (objects.size()/OBJ_COUNT_SCALE) * memoized_bbox.half_surface_area()) {
            // the split cost is more than not splitting at all so we are a leaf
            // The objects are already assigned to our object so nothing left to do for it
            left = right = nullptr;
        } else {
            ObjList left_objects,right_objects;
            for( int x=0; x<best_bin_split; x++){
                left_objects.insert(left_objects.end(),bins[x].objects.begin(),bins[x].objects.end());
            }
            for( int x=best_bin_split; x<NUM_BINS; x++){
                right_objects.insert(right_objects.end(),bins[x].objects.begin(),bins[x].objects.end());
            }
            objects.clear(); // no reason to hold onto the objects list since we will never check them
            left = new BVHList(left_objects,max_depth-1);
            right = new BVHList(right_objects,max_depth-1);
        }
    }
}

BVHList::~BVHList(){
    if(left) delete left;
    if(right) delete right;
}

BBox BVHList::bbox()const{
    return memoized_bbox;
}

bool BVHList::hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const{
    thread_local std::vector<std::pair<const BVHList*,RealRange>> stack;
    stack.clear();
    // Convenient lambda to check if a given intersection distance range is actaully a hit (as a bool)
    // Captures the allowed_distance which is modified in place by per-object hits
    auto hits_aabb_dists = [&allowed_distance](RealRange& int_dists){
        return
            int_dists.max >= int_dists.min && // Check if the ray direction actually would hit the AABB
            int_dists.max > allowed_distance.min && // And also need to check if the intersection is within our clipping range
            int_dists.min < allowed_distance.max;
    };

    // lets get the stack set up by adding in ourselves and then start walking down the tree
    stack.push_back({
        this,
        memoized_bbox.intersection_distance(ray),
    });

    bool found_hit = false;
    while(!stack.empty()) {
        const BVHList* next_to_check;
        RealRange int_dists;
        std::tie(next_to_check,int_dists) = stack.back();
        stack.pop_back();

        if(!hits_aabb_dists(int_dists)){
            //missed the box - try the next in the stack
            continue;
        }

        // Check if it is a leaf node and search its objects
        if (next_to_check->isLeaf()) {
            for(int x = 0; x < next_to_check->objects.size(); x++){
                found_hit |= next_to_check->objects[x]->hit(ray,allowed_distance,rec);
            }
            // Go to the next item in the stack - this leaf has no children to add to the stack
            continue;
        }

        // Not a leaf node, recurse down into until we find a leaf
        auto tleft  = next_to_check->left->memoized_bbox.intersection_distance(ray);
        auto tright = next_to_check->right->memoized_bbox.intersection_distance(ray);

        // check which AABB is closer to the ray orgin and leave that at the top of the stack
        if(tleft.min < tright.min){
            // Left is closer so put the right on before the left
            if( hits_aabb_dists(tright) ){
                stack.push_back({next_to_check->right,tright});
            }
            if( hits_aabb_dists(tleft) ){
                stack.push_back({next_to_check->left,tleft});
            }
        }else{
            if( hits_aabb_dists(tleft) ){
                stack.push_back({next_to_check->left,tleft});
            }
            if( hits_aabb_dists(tright) ){
                stack.push_back({next_to_check->right,tright});
            }
        }
    }
    return found_hit;
}

void BVHList::debug_print_tree(int)const{
    struct NodeInfo {
        const BVHList* node;
        int depth;
        size_t left_count;
        size_t right_count;
        double imbalance;
    };

    std::function<size_t(const BVHList*)> count_objects = [&](const BVHList* n) -> size_t {
        if(n->isLeaf()) return n->objects.size();
        return count_objects(n->left) + count_objects(n->right);
    };

    std::vector<NodeInfo> nodes;
    std::vector<std::pair<const BVHList*,int>> stack;
    stack.push_back({this, 0});
    while(!stack.empty()){
        auto [node, depth] = stack.back();
        stack.pop_back();
        if(node->isLeaf()) continue;
        size_t lc = count_objects(node->left);
        size_t rc = count_objects(node->right);
        double ratio = (double)std::max(lc,rc) / (double)std::max((size_t)1, std::min(lc,rc));
        nodes.push_back({node, depth, lc, rc, ratio});
        stack.push_back({node->left,  depth+1});
        stack.push_back({node->right, depth+1});
    }

    std::sort(nodes.begin(), nodes.end(), [](const NodeInfo& a, const NodeInfo& b){
        return a.imbalance > b.imbalance;
    });

    int to_print = std::min((int)nodes.size(), 50);
    printf("Top %d most unbalanced nodes (of %lu internal nodes):\n", to_print, nodes.size());
    for(int i=0; i<to_print; i++){
        auto& n = nodes[i];
        printf("  depth=%-3d  left=%-6lu  right=%-6lu  ratio=%.2fx\n",
            n.depth, n.left_count, n.right_count, n.imbalance);
    }
}

bool BVHList::isLeaf()const{
    // We can assume that leaf nodes will have no neighbor in left or right, and non-leaf nodes will have
    // both left and right due to how the constructor works.
    return !left;
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
int BVH8List::aabb_test_8(const BVH8Node& node,
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

void BVH8List::debug_print_tree(int) const {
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
}