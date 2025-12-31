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

    std::pair<ObjList, ObjList> minimal_surface_area_split(ObjList& dividing_objects, BBox& left, BBox& right);

    public:
    BVHList(const BVHList& other) = delete;
    BVHList(ObjList& world_objects,int max_depth = 25);
    ~BVHList();
    bool hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const;
    BBox bbox()const;
    bool isLeaf()const;
};