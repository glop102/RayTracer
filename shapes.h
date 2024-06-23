#pragma once
#include "vec_utils.h"
#include "utils.h"
#include "materials.h"

class Material;

struct HitRecord{
    Point3 intersection_point;
    //a scale of how far against the direction of the ray for the hit
    // ie the direction of a ray may not have been normalized and this is a scale factor on that length of the direction vector
    double distanceScale;
    Vector3 normal;
    bool front_face;
    std::shared_ptr<Material> material;
};


class Hittable{
    public:
    virtual ~Hittable() = default;
    virtual bool hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const = 0;
    virtual Vector3 min() const = 0;
    virtual Vector3 max() const = 0;
};

class Triangle:public Hittable{
    public:
    Point3 p1,p2,p3;
    Vector3 normal;
    std::shared_ptr<Material> material;
    Triangle(Point3 p1, Point3 p2, Point3 p3);
    Triangle(Point3 p1, Point3 p2, Point3 p3, std::shared_ptr<Material> mat);

    bool hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const;
    Vector3 min()const;
    Vector3 max()const;
};

class Sphere:public Hittable{
    public:
    Point3 center;
    double radius;
    std::shared_ptr<Material> material;
    Sphere(Point3 center, double radius);
    Sphere(Point3 center, double radius, std::shared_ptr<Material> mat);

    bool hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const;
    Vector3 min()const;
    Vector3 max()const;
};
