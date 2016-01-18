#include "Triangle.h"

Triangle::Triangle()
{
}

Triangle::Triangle(const Vec3& vec1, const Vec3& vec2, const Vec3& vec3,
                   const Texture& texture)
    :Shape(texture), v1(vec1), v2(vec2), v3(vec3)
{
    // this information will be used for optimizations in GetIntersect
    center = (v1 + v2 + v3) / 3.0;
    furthest = max(dist(v1, center), max(dist(v2, center), dist(v3, center)));
    
    Vec3 v21 = v2 - v1;
    Vec3 v32 = v3 - v2;
    
    // v1 is a point in the plane
    // the normal is the cross of the vector from v1 to v2 and v2 to v3
    plane = Plane(v1, cross(v21, v32).normalize(), texture);
}

double Triangle::GetIntersect(const Ray& ray)
{
    // if the ray does not intersect the plane that goes through
    // the triangle, it cannot possibly intersect the triangle
    double lambda;
    if((lambda = plane.GetIntersect(ray)) < 0)
    {
        return -1;
    }
    
    // we determine if the point where the ray intersects the
    // plane is wihin the boundaries of the triangle
    Vec3 point = ray.GetVec(lambda);
    if(dist(point, center) > furthest)
    {
        return -1;
    }
    
    Vec3 to_v1 = (v1 - point).normalize();
    Vec3 to_v2 = (v2 - point).normalize();
    Vec3 to_v3 = (v3 - point).normalize();
    
    double angle = 0;
    
    angle += acos(dot(to_v1, to_v2));
    angle += acos(dot(to_v2, to_v3));
    angle += acos(dot(to_v3, to_v1));
    
    if(fabs(2 * M_PI - angle) < 1e-6)
    {
        return lambda;
    }
    
    return -1;
}

const Vec3& Triangle::GetNormal(const Vec3& pos)
{
    return plane.GetNormal(pos);
}
