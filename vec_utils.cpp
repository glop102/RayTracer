#include "vec_utils.h"
#include <utility>
#include <cmath>

using std::sqrt;

const Vector3 x_pos={1.0,0.0,0.0};
const Vector3 y_pos={0.0,1.0,0.0};
const Vector3 z_pos={0.0,0.0,1.0};

const Vector3 x_neg={-1.0,0.0,0.0};
const Vector3 y_neg={0.0,-1.0,0.0};
const Vector3 z_neg={0.0,0.0,-1.0};

//===================================================================
// Vector3
//===================================================================
Vector3 Vector3::operator+(const Vector3 &other)const{
    return {
        this->x + other.x,
        this->y + other.y,
        this->z + other.z
        };
}
Vector3 Vector3::operator-(const Vector3 &other)const{
    return {
        this->x - other.x,
        this->y - other.y,
        this->z - other.z
        };
}
Vector3 Vector3::operator*(const Vector3& other)const{
    return {
        this->x * other.x,
        this->y * other.y,
        this->z * other.z
        };
}
Vector3 Vector3::operator/(const Vector3& other)const{
    return {
        this->x / other.x,
        this->y / other.y,
        this->z / other.z
        };
}
Vector3 Vector3::operator*(double scale)const{
    return {
        this->x * scale,
        this->y * scale,
        this->z * scale
        };
}
Vector3 Vector3::operator/(double scale)const{
    return {
        this->x / scale,
        this->y / scale,
        this->z / scale
        };
}

Vector3& Vector3::operator+=(const Vector3 &other){
    this->x += other.x;
    this->y += other.y;
    this->z += other.z;
    return *this;
}
Vector3& Vector3::operator-=(const Vector3 &other){
    this->x -= other.x;
    this->y -= other.y;
    this->z -= other.z;
    return *this;
}
Vector3& Vector3::operator*=(double scale){
    this->x *= scale;
    this->y *= scale;
    this->z *= scale;
    return *this;
}
Vector3& Vector3::operator/=(double scale){
    this->x /= scale;
    this->y /= scale;
    this->z /= scale;
    return *this;
}
double Vector3::operator[](const int idx)const{
    return data[idx];
}
double& Vector3::operator[](const int idx){
    return data[idx];
}
double Vector3::dot(const Vector3& other)const{
    return (this->x * other.x)
        + (this->y * other.y)
        + (this->z * other.z)
        ;
}
Vector3 Vector3::cross(const Vector3& other)const{
    Vector3 comb = {
        (this->y * other.z) - (this->z * other.y),
        (this->z * other.x) - (this->x * other.z),
        (this->x * other.y) - (this->y * other.x)
    };
    return std::move(comb);
}
double Vector3::length()const{
    return sqrt(length_squared());
}
double Vector3::length_squared()const{
    return (x*x)+(y*y)+(z*z);
}
Vector3 Vector3::normalize()const{
    double&& len = this->length();
    return {
        x / len,
        y / len,
        z / len
    };
}
Vector3 Vector3::unit_length()const{
    return std::move(this->operator/(this->length()));
}
Vector3 Vector3::reverse()const{
    return {
        -x,-y,-z
    };
}

Vector3 Vector3::random(){
    return {
        random_percentage_distribution(gen),
        random_percentage_distribution(gen),
        random_percentage_distribution(gen)
    };
}
Vector3 Vector3::random(double min, double max){
    double range = max-min;
    return {
        random_percentage_distribution(gen)*range + min,
        random_percentage_distribution(gen)*range + min,
        random_percentage_distribution(gen)*range + min
    };
}
Vector3 Vector3::random_unit_vector(){
    // Lets assume a 0,1,0 vector to start with, and we will rotate around the z axis
    // x = x*cosx - y*sinx => -sinx
    // y = x*sinx + y*cosx => cosx
    // z = z * 1 => 0
    //
    // Next we want to rotate around either the x or y axis
    // I don't think it matters for which we do, so I am picking the x axis
    // x = x => -sinx
    // y = y*cosy - z*siny = y*cosy => cosx*cosy
    // z = y*siny + z*cosy = y*siny => cosx*siny
    //
    // For completeness, I am going to also calculate the rotation around the y axis to see if the math looks equivilent
    // x = x*cosy + z*siny = x*cosy => -sinx*cosy
    // y = y => cosx
    // z = z*cosy - x*siny = x*siny => -sinx*siny
    // so just due arbitrary personal preference, I like the second rotation to be around the x axis

    // double rotx = random_percentage_distribution(gen)*2.0*PI;
    // double roty = random_percentage_distribution(gen)*2.0*PI;
    // double siny,sinx,cosy,cosx;
    // sincos(rotx,&sinx,&cosx);
    // sincos(roty,&siny,&cosy);
    // return {
    //     -sinx,
    //     cosx*cosy,
    //     cosx*siny
    // };

    // This is an alternative method that randomly attempts to generate one and then check its length
    // It is slightly faster than the above - on 10 million runs, ~477ms versus ~484ms
    // while(true){
    //     Vector3 v = Vector3::random(-1,1);
    //     double len_sq = v.length_squared();
    //     if(len_sq<1.0){
    //         return v / sqrt(len_sq);
    //     }
    // }

    // Using tan as a correction factor for picking a random point within a box should make it an even distribution
    return Vector3{
        tan(random_neg_pos_one(gen)),
        tan(random_neg_pos_one(gen)),
        tan(random_neg_pos_one(gen))
    }.normalize();
}

Vector3 Vector3::random_vector_on_hemisphere(const Vector3& normal){
    Vector3 v = random_unit_vector();
    if(v.dot(normal)<=0.0)
        return v.reverse();
    return std::move(v);
}

Vector3 Vector3::lerp(const Vector3& first, const Vector3& second, double scale){
    return (first* (1.0-scale) ) + second*scale;
}

void Vector3::min_accum(Vector3& accum, const Vector3& val){
    if(val.x<accum.x) accum.x = val.x;
    if(val.y<accum.y) accum.y = val.y;
    if(val.z<accum.z) accum.z = val.z;
}
void Vector3::max_accum(Vector3& accum, const Vector3& val){
    if(val.x>accum.x) accum.x = val.x;
    if(val.y>accum.y) accum.y = val.y;
    if(val.z>accum.z) accum.z = val.z;
}

Vector3 Vector3::reflect_around_normal(const Vector3& normal, const Vector3& incoming){
    // https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
    double projection = 2 * incoming.dot(normal);
    return incoming - (normal * projection);
}
Vector3 Vector3::reflect(const Vector3& incoming)const{
    return reflect_around_normal(*this,incoming);
}

Vector3 Vector3::refract_around_normal(const Vector3& normal, const Vector3& incoming, const double& refractive_index_ratio){
    auto cos_theta = fmin(1.0, incoming.reverse().dot(normal));
    Vector3 r_out_perp =  (incoming + (normal*cos_theta)) * refractive_index_ratio;
    Vector3 r_out_parallel = normal * -sqrt(fabs(1.0 - r_out_perp.length_squared()));
    return r_out_perp + r_out_parallel;

    // Slower validation code
    // auto incoming_theta = acos(cos_theta);
    // auto outgoing_theta = incoming_theta * refractive_index_ratio;
    // auto diff_theta = incoming_theta - outgoing_theta;
    // auto rot_axis = normal.cross(incoming);
    // return rot_axis.rotate(incoming,diff_theta);
}
Vector3 Vector3::refract(const Vector3& incoming, const double& refractive_index_ratio)const{
    return refract_around_normal(*this,incoming,refractive_index_ratio);
}

Vector3 Vector3::rotate(const Vector3& point, double radians)const{
    // https://suricrasia.online/blog/shader-functions/
    double sinrot,cosrot;
    sincos(radians,&sinrot,&cosrot);

    Vector3 projection = operator*( dot(point) );
    return lerp(projection, point, cosrot) + (cross(point) * sinrot);
}

bool Vector3::near_zero()const{
    return
        std::fabs(x) < std::numeric_limits<double>::epsilon()*15.0 &&
        std::fabs(y) < std::numeric_limits<double>::epsilon()*15.0 &&
        std::fabs(z) < std::numeric_limits<double>::epsilon()*15.0;
}

//===================================================================
// Ray
//===================================================================
Vector3 Ray::at(double distanceScale)const{
    return this->direction * distanceScale + this->origin;
}
void Ray::debug_print()const{
    print("{:.2f} {:.2f} {:.2f}->",origin.x,origin.y,origin.z);
    print("{:.2f} {:.2f} {:.2f}  ",direction.x,direction.y,direction.z);
}


//===================================================================
// BBox
//===================================================================

double BBox::half_surface_area()const{
    Vector3 dv = max-min;
    return (dv.x*(dv.z+dv.y))+(dv.z*dv.y);
}
RealRange BBox::intersection_distance(const Ray& ray)const{
    // Taken from https://tavianator.com/2011/ray_box.html
    // It is a specific impl of the SLAB method which bounds the direction in
    // for every plane, and then finds the farthest/nearest intersections for the planes

    // MAX and MIN start at the top here meaning absolute distance from the origin for the corners of the AABB
    // dsmin and dsmax are then sorting the near/far for time-of-flight distances
    auto dmin = (min - ray.origin)/ray.direction;
    auto dmax = (max - ray.origin)/ray.direction;

    Vector3 dsmin,dsmax; // sorted versions ie min is the min of both axes
    for(int i=0; i<3; i++){
        dsmin[i] = std::min(dmin[i],dmax[i]);
        dsmax[i] = std::max(dmin[i],dmax[i]);
    }

    // Now that we know the distances for intersections for x/y/z, we can tell if the
    // intersections on each plane makes sense for our ray
    return {
        std::max(std::max(dsmin[0],dsmin[1]),dsmin[2]),
        std::min(std::min(dsmax[0],dsmax[1]),dsmax[2]),
    };
}

void BBox::absorb(const BBox& other) {
    Vector3::min_accum(this->min,other.min);
    Vector3::max_accum(this->max,other.max);
}
void BBox::absorb(const Point3& point) {
    Vector3::min_accum(this->min,point);
    Vector3::max_accum(this->max,point);
}

Point3 BBox::center()const{
    return {
        (min.x + max.x) / 2.0,
        (min.y + max.y) / 2.0,
        (min.z + max.z) / 2.0,
    };
}