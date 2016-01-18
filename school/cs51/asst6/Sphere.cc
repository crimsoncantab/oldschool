#include "Sphere.h"

Sphere::Sphere()
{
}

Sphere::Sphere(const Vec3& center, double radius, const Texture& texture)
    :Shape(texture), cen(center), rad(radius)
{
}

double Sphere::GetIntersect(const Ray& ray)
{
    Vec3 rayCenter = cen - ray.GetPoint();
    
    double proj = dot(rayCenter, ray.GetDir());
    if(proj < 0)
    {
        return -1;
    }
    
    double tmp = rad * rad + proj * proj;
    tmp -= dot(rayCenter, rayCenter);
    
    if(tmp < 0)
    {
        return -1;
    }
    
    return proj - sqrt(tmp);
}

const Vec3& Sphere::GetNormal(const Vec3& pos)
{
    norm = (pos - cen).normalize();
    return norm;
}
