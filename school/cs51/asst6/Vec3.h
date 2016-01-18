#ifndef _VEC3_
#define _VEC3_

#include <cmath>
#include <iostream>

class Vec3
{
 private:
    double x, y, z;
    
 public:
    Vec3();
    Vec3(double x, double y, double z);
    
    double GetX() const { return x; }
    double GetY() const { return y; }
    double GetZ() const { return z; }
    
    Vec3 operator += (const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    Vec3 operator + (const Vec3& v) const { return Vec3(*this) += v; }
    
    Vec3 operator -= (const Vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    Vec3 operator - (const Vec3& v) const { return Vec3(*this) -= v; }
    
    Vec3 operator *= (const double s) { x *= s; y *= s; z *= s;	return *this; }
    Vec3 operator * (const double s) const { return Vec3(*this) *= s; }
    
    Vec3 operator /= (const double s) { x /= s; y /= s; z /= s;	return *this; }
    Vec3 operator / (const double s) const { return Vec3(*this) /= s; }
    
    Vec3 normalize();
};

Vec3 operator * (double s, const Vec3& v);
std::ostream& operator <<(std::ostream &os, const Vec3 &v);

Vec3 cross(const Vec3& v1, const Vec3& v2);
Vec3 clamp(const Vec3& v, double min, double max);
Vec3 reflect(const Vec3& incoming, const Vec3& normal);
double dot(const Vec3& v1, const Vec3& v2);
double dist(const Vec3& v1, const Vec3& v2);

#endif //_VEC3_
