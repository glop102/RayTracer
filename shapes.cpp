#include "shapes.h"
#include <cmath>
using std::sqrt;

//===================================================================
// Triangle
//===================================================================
Triangle::Triangle(const Point3& p1, const Point3& p2, const Point3& p3):
    p1(p1), p2(p2), p3(p3), material(AluminiumDull)
{
    normal = (p2-p1).cross(p3-p1).normalize();
}
Triangle::Triangle(const Point3& p1, const Point3& p2, const Point3& p3, std::shared_ptr<Material> mat):
    p1(p1), p2(p2), p3(p3), material(mat)
{
    normal = (p2-p1).cross(p3-p1).normalize();
}

BBox Triangle::bbox()const{
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
    // https://courses.cs.washington.edu/courses/csep557/10au/lectures/triangle_intersection.pdf
    // The intersection test occurs in two stages
    // 1) Find the point of the ray cast on the plane of the triangle
    // 2) Determine if that point is inside the triangle

    auto normal_direction_dot = this->normal.dot(ray.direction);
    if (normal_direction_dot <= std::numeric_limits<double>::epsilon()*15.0 && normal_direction_dot >= std::numeric_limits<double>::epsilon()*-15.0) {
        // The direction is effectivly parrellel with our triangle so consider it a miss
        return false;
    }

    // There is a random coefficient d that shows up in the math, but it just is the dot product
    // of the normal with some point in the triangle's plane and so we can take the first point
    // of our triangle as that point in the plane.
    auto d = this->normal.dot(this->p1);

    // The distance the ray travels to rach our plane is d- n.dot(ray.origin) / n.dot(ray.direction)
    // We calculated most of it above so lets finish it out
    auto ray_intersection_distance = ( d - this->normal.dot(ray.origin) ) / normal_direction_dot;
    if(!allowed_distance.surrounds(ray_intersection_distance)) {
        return false;
    }
    auto ray_intersection_point = ray.at(ray_intersection_distance);

    // We know where the ray intersects the *plane* of the triangle but we do not know if that
    // intersection point is inside the triangle. Lets check that now.
    // Simply check if the normal with the point on the plane is the same direction for all 3 sides
    bool outside_side_a = (p2-p1).cross(ray_intersection_point-p1).dot(normal) < 0;
    bool outside_side_b = (p3-p2).cross(ray_intersection_point-p2).dot(normal) < 0;
    bool outside_side_c = (p1-p3).cross(ray_intersection_point-p3).dot(normal) < 0;
    if (outside_side_a || outside_side_b || outside_side_c) return false;

    // todo - barycentric coords to do texture mapping with
    // todo - per-vertex normals that get interpolated with the barycentric coords

    // We have passed all the tests for if the point is inside the triangle, so lets do some bookkeeping
    // Shrink the far plane for finding more hits that are only closer
    allowed_distance.max = ray_intersection_distance;
    // Add the hit information to the hit record
    rec.intersection_point = ray_intersection_point;
    rec.distanceScale = ray_intersection_distance;
    rec.material = this->material;
    // We want to know if we hit the front or back face of the triangle, which is just comparing if the
    // direction the ray is traveling is the same or opposite direction of the normal
    if (normal_direction_dot < 0.0) {
        rec.normal = this->normal;
        rec.front_face = true;
    } else {
        rec.normal = this->normal.reverse();
        rec.front_face = false;
        // rec.material = ErrorMaterialRed;
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