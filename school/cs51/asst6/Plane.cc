#include "Plane.h"

Plane::Plane()
{
}

Plane::Plane(const Vec3& point, const Vec3& normal, const Texture& texture)
    :Shape(texture), pt(point), norm(normal)
{
    norm.normalize();
}

double Plane::GetIntersect(const Ray& ray)
{
    double denom = dot(ray.GetDir(), norm);
    if(denom == 0)
    {
        return -1;
    }
    
    double d1 = dot(ray.GetPoint(), norm);
    double d2 = dot(pt, norm);
    double dist = (d2 - d1) / denom;
    if(dist < 0)
    {
        return -1;
    }
    return dist;
}

const Vec3& Plane::GetNormal(const Vec3& pos)
{
    return norm;
}
