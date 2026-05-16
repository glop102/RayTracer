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
    virtual BBox bbox() const = 0;
};

class Triangle:public Hittable{
    public:
    Point3 p1,p2,p3;
    Vector3 normal;
    std::shared_ptr<Material> material;
    Triangle(const Point3& p1, const Point3& p2, const Point3& p3);
    Triangle(const Point3& p1, const Point3& p2, const Point3& p3, std::shared_ptr<Material> mat);

    bool hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const;
    BBox bbox()const;
};

class Sphere:public Hittable{
    public:
    Point3 center;
    double radius;
    std::shared_ptr<Material> material;
    Sphere(const Point3& center, double radius);
    Sphere(const Point3& center, double radius, std::shared_ptr<Material> mat);

    bool hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const;
    BBox bbox()const;
};

std::vector<std::shared_ptr<Triangle>> make_cube(double radius, const Point3& center, std::shared_ptr<Material> material);