#pragma once
#include <utility>
#include "utils.h"

class Vector3{
    public:
    union{
        struct{
            double x,y,z;
        };
        struct{
            double red,green,blue;
        };
        double data[3];
    };
    
    Vector3 operator+(const Vector3& other)const;
    Vector3 operator-(const Vector3& other)const;
    Vector3 operator*(const Vector3& other)const;
    Vector3 operator/(const Vector3& other)const;
    Vector3 operator*(double scale)const;
    Vector3 operator/(double scale)const;
    Vector3& operator+=(const Vector3& other);
    Vector3& operator-=(const Vector3& other);
    Vector3& operator*=(double);
    Vector3& operator/=(double);
    double dot(const Vector3& other)const;
    Vector3 cross(const Vector3& other)const;
    double length()const;
    double length_squared()const;
    Vector3 normalize()const;
    Vector3 unit_length()const;

    static Vector3 random();
    static Vector3 random(double min, double max);
    static Vector3 random_unit_vector();
    static Vector3 random_vector_on_hemisphere(const Vector3& normal);
    static Vector3 lerp(const Vector3& first, const Vector3& second, double scale);
    static void min_accum(Vector3& accum, const Vector3& val);
    static void max_accum(Vector3& accum, const Vector3& val);

    static Vector3 reflect_around_normal(const Vector3& normal, const Vector3& incoming);
    Vector3 reflect(const Vector3& incoming)const;
    static Vector3 refract_around_normal(const Vector3& normal, const Vector3& incoming, const double& refractive_index_ratio);
    Vector3 refract(const Vector3& incoming, const double& refractive_index_ratio)const;
    // rotate a point around this vector as the rotation axis
    Vector3 rotate(const Vector3& point, double radians)const;

    bool near_zero()const;
};
using Point3 = Vector3;

class Ray{
    public:
    Point3 origin;
    Vector3 direction;
    Vector3 at(double distanceScale)const;
    void debug_print()const;
};

extern const Vector3 x_pos,y_pos,z_pos;
extern const Vector3 x_neg,y_neg,z_neg;