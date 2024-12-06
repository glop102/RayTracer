#pragma once
#include <vector>
#include <memory>
#include "shapes.h"
#include "utils.h"

class HittableList:public Hittable{
    public:
    std::vector<std::shared_ptr<Hittable>> objects;
    bool hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const;
    void add(std::shared_ptr<Hittable> object);
    void clear();
    BBox bbox()const;
};

class BVHList:public Hittable{
    public:
    using ObjList = std::vector<std::shared_ptr<Hittable>>;
    protected:
    BVHList *left, *right;
    ObjList objects;
    int max_depth_allowed; // how much more depth is allowed
    BBox memoized_bbox;

    std::pair<ObjList, ObjList> minimal_surface_area_split(ObjList& dividing_objects, BBox& left, BBox& right);

    //For efficency, we want to do hit calculations differently for the first box versus sub-boxes
    // ie we want to recurse to closer sub-boxes before further boxes and so we need to test both to pick the right one instead of blindly recursing
    bool hit_internal(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const;

    public:
    BVHList(const BVHList& other) = delete;
    BVHList(ObjList& world_objects,int max_depth = 25);
    ~BVHList();
    bool hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const;
    BBox bbox()const;
};