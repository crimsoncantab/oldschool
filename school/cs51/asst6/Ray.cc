#include "Ray.h"

Ray::Ray(const Vec3& point, const Vec3& direction)
    :pt(point), dir(direction)
{
    dir.normalize();
}

const Vec3& Ray::GetPoint() const
{
    return pt;
}

const Vec3& Ray::GetDir() const
{
    return dir;
}

const Vec3 Ray::GetVec(double lambda) const
{
    return pt + lambda * dir;
}
