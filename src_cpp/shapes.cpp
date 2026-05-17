#include "shapes.h"
#include <cmath>
using std::sqrt;

//===================================================================
// Triangle
//===================================================================
Triangle::Triangle(const Point3& p1, const Point3& p2, const Point3& p3):
    p1(p1), e1(p2-p1), e2(p3-p1), material(AluminiumDull)
{
    normal = e1.cross(e2).normalize();
}
Triangle::Triangle(const Point3& p1, const Point3& p2, const Point3& p3, std::shared_ptr<Material> mat):
    p1(p1), e1(p2-p1), e2(p3-p1), material(mat)
{
    normal = e1.cross(e2).normalize();
}

BBox Triangle::bbox()const{
    Point3 p2 = p1+e1, p3 = p1+e2;
    return {
        {
            std::min(std::min(p1.x,p2.x),p3.x),
            std::min(std::min(p1.y,p2.y),p3.y),
            std::min(std::min(p1.z,p2.z),p3.z)
        },
        {
            std::max(std::max(p1.x,p2.x),p3.x),
            std::max(std::max(p1.y,p2.y),p3.y),
            std::max(std::max(p1.z,p2.z),p3.z)
        }
    };
}

bool Triangle::hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const{
    // Möller-Trumbore algorithm
    // Solves origin + t*dir = p1 + u*e1 + v*e2 simultaneously for t, u, v.
    // u and v are barycentric coordinates: the point is inside the triangle if
    // u >= 0, v >= 0, and u+v <= 1.

    Vector3 h = ray.direction.cross(e2);
    double a = e1.dot(h);
    // a is zero when the ray is parallel to the triangle plane
    if (a > -1e-8 && a < 1e-8) return false;

    double f = 1.0 / a;
    Vector3 s = ray.origin - p1;
    double u = f * s.dot(h);
    if (u < 0.0 || u > 1.0) return false;  // outside triangle, early exit

    Vector3 q = s.cross(e1);
    double v = f * ray.direction.dot(q);
    if (v < 0.0 || u + v > 1.0) return false;  // outside triangle, early exit

    double t = f * e2.dot(q);
    if (!allowed_distance.surrounds(t)) return false;

    allowed_distance.max = t;
    rec.distanceScale = t;
    rec.intersection_point = ray.at(t);
    rec.material = material;
    // a > 0 means ray hits the front face (opposite sign to normal.dot(ray.direction))
    if (a > 0.0) {
        rec.normal = normal;
        rec.front_face = true;
    } else {
        rec.normal = normal.reverse();
        rec.front_face = false;
    }
    return true;
}

//===================================================================
// Sphere
//===================================================================
Sphere::Sphere(const Point3& center, double radius):
    center(center),radius(radius),material(AluminiumDull)
{}
Sphere::Sphere(const Point3& center, double radius,std::shared_ptr<Material> mat):
    center(center),radius(radius),material(mat)
{}

BBox Sphere::bbox()const{
    return {
        {
            center.x - radius,
            center.y - radius,
            center.z - radius
        },
        {
            center.x + radius,
            center.y + radius,
            center.z + radius
        }
    };
}

bool Sphere::hit(const Ray& ray, RealRange& allowed_distance, HitRecord& rec)const{
    // Cheaper bounding box check to weed out any misses early
    auto t = this->bbox().intersection_distance(ray);
    if(t.min > t.max || t.min > allowed_distance.max || t.max < allowed_distance.min)
        return false;

    Vector3 oc = center - ray.origin;
    auto a = ray.direction.length_squared();
    auto h = ray.direction.dot(oc);
    auto c = oc.length_squared() - (radius*radius);
    auto discriminant = h*h - a*c;
    if(discriminant<0.0){
        return false;
    }
    auto sqrtd = sqrt(discriminant);

    auto root = (h-sqrtd)/a;
    if(!allowed_distance.surrounds(root)){
        root = (h+sqrtd)/a;
        if(!allowed_distance.surrounds(root)){
            return false;
        }
    }

    // shrink the far plane to keep ensuring we only get closer hits for any further entities found
    allowed_distance.max = root;

    rec.distanceScale = root;
    rec.intersection_point = ray.at(root);
    rec.material = material;
    rec.normal = (rec.intersection_point - center) / radius;
    if(ray.direction.dot(rec.normal)>0.0){
        rec.front_face = false;
        rec.normal = rec.normal.reverse();
    }else{
        rec.front_face = true;
    }
    return true;
}


//===================================================================
//  Utilities
//===================================================================
std::vector<std::shared_ptr<Triangle>> make_cube(double radius, const Point3& center, std::shared_ptr<Material> material) {
    /*
    We want counter-clockwise on every face
    --,+-,++,-+ if thinking abount the corners
    */
    // This is the set of points for the "front" face of the cube
    // Specifically the closest face when looking at the origin from the positive x axis direction
    // this is wound in the counter-clockwise order
    const Vector3 front_face_og[4] = {
        {1,-1,-1},
        {1,1,-1},
        {1,1,1},
        {1,-1,1},
    };
    // And this is the set of points for the back face, also in counter-clockwise order
    const Vector3 back_face_og[4] = {
        {-1,-1,-1},
        {-1,-1,1},
        {-1,1,1},
        {-1,1,-1},
    };
    Vector3 front_face[4] = {
        front_face_og[0]*radius + center,
        front_face_og[1]*radius + center,
        front_face_og[2]*radius + center,
        front_face_og[3]*radius + center,
    };
    Vector3 back_face[4] = {
        back_face_og[0]*radius + center,
        back_face_og[1]*radius + center,
        back_face_og[2]*radius + center,
        back_face_og[3]*radius + center,
    };
    return {
        // Front side of box
        std::make_shared<Triangle>(
            front_face[0],
            front_face[1],
            front_face[2],
            material
        ),
        std::make_shared<Triangle>(
            front_face[2],
            front_face[3],
            front_face[0],
            material
        ),
        // Back side of box
        std::make_shared<Triangle>(
            back_face[0],
            back_face[1],
            back_face[2],
            material
        ),
        std::make_shared<Triangle>(
            back_face[2],
            back_face[3],
            back_face[0],
            material
        ),
        // Top side of box
        std::make_shared<Triangle>(
            front_face[2],
            front_face[1],
            back_face[3],
            material
        ),
        std::make_shared<Triangle>(
            back_face[3],
            back_face[2],
            front_face[2],
            material
        ),
        // Bottom side of box
        std::make_shared<Triangle>(
            front_face[0],
            front_face[3],
            back_face[1],
            material
        ),
        std::make_shared<Triangle>(
            back_face[1],
            back_face[0],
            front_face[0],
            material
        ),
        // Right Side of the box
        std::make_shared<Triangle>(
            front_face[3],
            front_face[2],
            back_face[2],
            material
        ),
        std::make_shared<Triangle>(
            back_face[2],
            back_face[1],
            front_face[3],
            material
        ),
        // Left Side of the box
        std::make_shared<Triangle>(
            back_face[0],
            back_face[3],
            front_face[1],
            material
        ),
        std::make_shared<Triangle>(
            front_face[1],
            front_face[0],
            back_face[0],
            material
        ),
    };
}