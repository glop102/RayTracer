#pragma once
#include <vector>
#include <memory>
#include <cstdint>
#include "shapes.h"
#include "utils.h"

using ObjList = std::vector<std::shared_ptr<Hittable>>;

struct alignas(32) BVH8Node {
    float min_x[8], min_y[8], min_z[8];
    float max_x[8], max_y[8], max_z[8];
    int32_t child[8];
    uint8_t num_children;
};

class HittableList:public Hittable{
    public:
    ObjList objects;
    bool hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const;
    void add(std::shared_ptr<Hittable> object);
    void clear();
    BBox bbox()const;
};


class BVH8List : public Hittable {
    struct Leaf { int obj_start, obj_count; };

    std::vector<BVH8Node> nodes;
    std::vector<Leaf>     leaves;
    ObjList               flat_objects;
    BBox                  root_bbox;

    void build(int node_idx, ObjList objs, int depth);
    std::pair<ObjList,ObjList> sah_split(ObjList& objs);
    public:
    BVH8List(const BVH8List&) = delete;
    BVH8List(ObjList& world_objects, int max_depth = 8);
    bool hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec) const;
    BBox bbox() const;
    void debug_print_tree(int = 0) const;
};

class TriangleBVH8 : public Hittable {
    struct Leaf { int obj_start, obj_count; };

    std::vector<BVH8Node>  nodes;
    std::vector<Leaf>      leaves;
    std::vector<Triangle>  flat_triangles;
    BBox                   root_bbox;

    void build(int node_idx, std::vector<Triangle> tris, int depth);
    std::pair<std::vector<Triangle>, std::vector<Triangle>> sah_split(std::vector<Triangle>& tris);
    public:
    TriangleBVH8(const TriangleBVH8&) = delete;
    TriangleBVH8(std::vector<Triangle>& triangles, int max_depth = 8);
    bool hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec) const;
    BBox bbox() const;
    void debug_print_tree(int = 0) const;
};


// An Instance wraps a BVH8List with a rigid-body world transform (rotation + translation).
// Build the geometry in object space, then rotate_y / translate to place it in the scene.
// Transforms compose left-to-right: rotate first, then translate.
class Instance : public Hittable {
    std::shared_ptr<Hittable> bvh;
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
    void debug_print_tree(int indent=0) const override;
};