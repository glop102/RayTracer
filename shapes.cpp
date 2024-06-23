#include "shapes.h"
#include <cmath>
using std::sqrt;

//===================================================================
// Triangle
//===================================================================
Triangle::Triangle(Point3 p1, Point3 p2, Point3 p3):
    p1(p1), p2(p2), p3(p3), material(AluminiumDull)
{
    normal = (p2-p1).cross(p3-p1).normalize();
}
Triangle::Triangle(Point3 p1, Point3 p2, Point3 p3, std::shared_ptr<Material> mat):
    p1(p1), p2(p2), p3(p3), material(mat)
{
    normal = (p2-p1).cross(p3-p1).normalize();
}

Vector3 Triangle::min()const{
    return {
        std::min(std::min(p1.x,p2.x),p3.x),
        std::min(std::min(p1.y,p2.y),p3.y),
        std::min(std::min(p1.z,p2.z),p3.z)
    };
}
Vector3 Triangle::max()const{
    return {
        std::max(std::max(p1.x,p2.x),p3.x),
        std::max(std::max(p1.y,p2.y),p3.y),
        std::max(std::max(p1.z,p2.z),p3.z)
    };
}

bool Triangle::hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const{
    return false;
}

//===================================================================
// Sphere
//===================================================================
Sphere::Sphere(Point3 center, double radius):
    center(center),radius(radius),material(AluminiumDull)
{}
Sphere::Sphere(Point3 center, double radius,std::shared_ptr<Material> mat):
    center(center),radius(radius),material(mat)
{}

Vector3 Sphere::min()const{
    return {
        center.x - radius,
        center.y - radius,
        center.z - radius
    };
}
Vector3 Sphere::max()const{
    return {
        center.x + radius,
        center.y + radius,
        center.z + radius
    };
}

bool Sphere::hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const{
    Vector3 oc = center - ray.origin;
    auto a = ray.direction.length_squared();
    auto h = ray.direction.dot(oc);
    auto c = oc.length_squared() - (radius*radius);
    auto discriminant = h*h - a*c;
    if(discriminant<0.0){
        return false;
    }
    auto sqrtd = sqrt(discriminant);
    //attempt to make it branchless - failure, but an attempt
    auto root_sub = (h-sqrtd)/a;
    auto root_add = (h+sqrtd)/a;
    bool root_sub_valid = allowed_distance.surrounds(root_sub);
    bool root_add_valid = allowed_distance.surrounds(root_add);
    //Check that the intersection is within our bounds
    if(!root_sub_valid && !root_add_valid){
        return false;
    }
    auto root = root_sub ? root_sub : root_add;
    // double root = root_sub*root_sub_valid;

    // auto root = (h-sqrtd)/a;
    // if(!allowed_distance.surrounds(root)){
    //     root = (h+sqrtd)/a;
    //     if(!allowed_distance.surrounds(root)){
    //         return false;
    //     }
    // }

    // shrink the far plane to keep ensuring we only get closer hits for any further entities found
    allowed_distance.max = root;

    rec.distanceScale = root;
    rec.intersection_point = ray.at(root);
    rec.material = material;
    rec.normal = (rec.intersection_point - center) / radius;
    if(ray.direction.dot(rec.normal)>0.0){
        rec.front_face = false;
        rec.normal *= -1.0;
    }else{
        rec.front_face = true;
    }
    return true;
}