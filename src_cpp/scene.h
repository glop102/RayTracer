#pragma once
#include <vector>
#include <memory>
#include "shapes.h"
#include "utils.h"

using ObjList = std::vector<std::shared_ptr<Hittable>>;

class HittableList:public Hittable{
    public:
    ObjList objects;
    bool hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const;
    void add(std::shared_ptr<Hittable> object);
    void clear();
    BBox bbox()const;
};

class BVHList:public Hittable{
    protected:
    BVHList *left, *right;
    ObjList objects;
    int max_depth_allowed; // how much more depth is allowed
    BBox memoized_bbox;

    public:
    BVHList(const BVHList& other) = delete;
    BVHList(ObjList& world_objects,int max_depth = 25);
    ~BVHList();
    bool hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const;
    BBox bbox()const;
    bool isLeaf()const;
    void debug_print_tree(int indent=0)const;
};


class BVH8List : public Hittable {
    struct alignas(32) BVH8Node {
        float min_x[8], min_y[8], min_z[8];  // SoA: all 8 children's bounds by axis
        float max_x[8], max_y[8], max_z[8];  // 32-byte aligned for AVX2 _mm256_load_ps
        int32_t child[8];   // >= 0: node index, < 0: ~index into leaves[]
        uint8_t num_children;
    };
    struct Leaf { int obj_start, obj_count; };

    std::vector<BVH8Node> nodes;
    std::vector<Leaf>     leaves;
    ObjList               flat_objects;
    BBox                  root_bbox;

    void build(int node_idx, ObjList objs, int depth);
    std::pair<ObjList,ObjList> sah_split(ObjList& objs);
    static int aabb_test_8(const BVH8Node& node,
                           float ox, float oy, float oz,
                           float idx, float idy, float idz,
                           float t_near, float t_far,
                           float* tmin_out);
    public:
    BVH8List(const BVH8List&) = delete;
    BVH8List(ObjList& world_objects, int max_depth = 8);
    bool hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec) const;
    BBox bbox() const;
    void debug_print_tree(int = 0) const;
};


// An Instance wraps a BVH8List with a rigid-body world transform (rotation + translation).
// Build the geometry in object space, then rotate_y / translate to place it in the scene.
// Transforms compose left-to-right: rotate first, then translate.
class Instance : public Hittable {
    std::shared_ptr<BVH8List> bvh;
    double R[3][3];      // rotation: object → world
    Vector3 translation; // world-space offset applied after rotation
    BBox memoized_bbox;

    Vector3 rot(const Vector3& v) const;     // apply R  (obj → world)
    Vector3 rot_inv(const Vector3& v) const; // apply R^T (world → obj)
    void recompute_bbox();

public:
    Instance(ObjList& objects, int max_depth = 8);
    Instance& rotate_y(double degrees);
    Instance& translate(const Vector3& offset);
    bool hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec) const override;
    BBox bbox() const override;
};