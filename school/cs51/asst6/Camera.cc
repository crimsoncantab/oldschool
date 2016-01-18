#include "Camera.h"	

Camera::Camera()
{
}

Camera::Camera(const Vec3& position, const Vec3& lookat)
    :pos(position)
{
    forward = (lookat - pos).normalize();
    right = cross(Vec3(0, -1, 0), forward).normalize();
    up = cross(right, forward).normalize();
}

Ray Camera::Shoot(double x, double y) const
{
    Vec3 dir = (forward + (x * right) + (y * up)).normalize();
    return Ray(pos, dir);
}
