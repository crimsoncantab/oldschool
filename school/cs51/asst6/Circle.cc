#include "Circle.h"

Circle::Circle()
{
}

Circle::Circle(const Vec3& c, const Vec3& n, const double r,
               const Texture& texture)
    :Shape(texture), center(c)
{
    plane = Plane(c, n, texture);
    radius = r;
}

double Circle::GetIntersect(const Ray& ray)
{
    double lambda;
    //checks if plane intersects
    if ((lambda = plane.GetIntersect(ray)) < 0)
        return -1;

    Vec3 point = ray.GetVec(lambda);
    //if it intersects plane, check if distance from center is within radius
    if (dist(point, center) <= radius)
        return lambda;

    return -1;
}

const Vec3& Circle::GetNormal(const Vec3& pos)
{
    //use plane to fine normal
    return plane.GetNormal(pos);
}
