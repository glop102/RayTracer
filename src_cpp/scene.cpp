#include "scene.h"
#include <utility>
#include <functional>

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