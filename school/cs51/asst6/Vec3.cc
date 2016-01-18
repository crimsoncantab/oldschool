#include "Vec3.h"

Vec3::Vec3()
    :x(0), y(0), z(0)
{
}

Vec3::Vec3(double x, double y, double z)
    :x(x), y(y), z(z)
{
}

Vec3 Vec3::normalize()
{
    double d = dot(*this, *this);
    return *this /= sqrt(d);
}

Vec3 operator * (double s, const Vec3& v)
{
    return v * s;
}

std::ostream& operator <<(std::ostream &os, const Vec3 &v)
{
    os << "<" << v.GetX() << " " << v.GetY() << " " << v.GetZ() << ">";
    return os;
}


Vec3 cross(const Vec3& v1, const Vec3& v2)
{
    double x = v1.GetY() * v2.GetZ() - v1.GetZ() * v2.GetY();
    double y = v1.GetZ() * v2.GetX() - v1.GetX() * v2.GetZ();
    double z = v1.GetX() * v2.GetY() - v1.GetY() * v2.GetX();
    return Vec3(x, y, z);
}

Vec3 clamp(const Vec3& v, double min, double max)
{
    double x = v.GetX() > max ? max : (v.GetX() < min ? min : v.GetX());
    double y = v.GetY() > max ? max : (v.GetY() < min ? min : v.GetY());
    double z = v.GetZ() > max ? max : (v.GetZ() < min ? min : v.GetZ());
    return Vec3(x, y, z);
}

Vec3 reflect(const Vec3& incoming, const Vec3& normal)
{
    return incoming - 2 * dot(incoming, normal) * normal;
}

double dot(const Vec3& v1, const Vec3& v2)
{
    double d = 0;
    d += v1.GetX() * v2.GetX();
    d += v1.GetY() * v2.GetY();
    d += v1.GetZ() * v2.GetZ();
    return d;
}

double dist(const Vec3& v1, const Vec3& v2)
{
    double d = 0;
    d += (v1.GetX() - v2.GetX()) * (v1.GetX() - v2.GetX());
    d += (v1.GetY() - v2.GetY()) * (v1.GetY() - v2.GetY());
    d += (v1.GetZ() - v2.GetZ()) * (v1.GetZ() - v2.GetZ());
    return sqrt(d);
}
