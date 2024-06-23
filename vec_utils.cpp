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
    return (this->x*this->x)+(this->y*this->y)+(this->z*this->z);
}
Vector3& Vector3::normalize(){
    double&& len = this->length();
    this->x /= len;
    this->y /= len;
    this->z /= len;
    return *this;
}
Vector3 Vector3::unit_length()const{
    return std::move(this->operator/(this->length()));
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
    while(true){
        Vector3 v = Vector3::random(-1,1);
        double len_sq = v.length_squared();
        if(len_sq<1.0){
            return v / sqrt(len_sq);
        }
    }

    // Using tan as a correction factor for picking a random point within a box should make it an even distribution
    // return Vector3{
    //     tan(random_neg_pos_one(gen)),
    //     tan(random_neg_pos_one(gen)),
    //     tan(random_neg_pos_one(gen))
    // }.normalize();
}

Vector3 Vector3::random_vector_on_hemisphere(const Vector3& normal){
    Vector3 v = random_unit_vector();
    if(v.dot(normal)<=0.0)
        return v*-1.0;
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

Vector3 Vector3::rotate(const Vector3& point, double radians)const{
    // https://suricrasia.online/blog/shader-functions/
    double sinrot,cosrot;
    sincos(radians,&sinrot,&cosrot);

    Vector3 projection = operator*( dot(point) );
    return lerp(projection, point, cosrot) + (cross(point) * sinrot);
}

bool Vector3::near_zero()const{
    return
        std::fabs(x) < std::numeric_limits<double>::epsilon()*3.0 &&
        std::fabs(y) < std::numeric_limits<double>::epsilon()*3.0 &&
        std::fabs(z) < std::numeric_limits<double>::epsilon()*3.0;
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