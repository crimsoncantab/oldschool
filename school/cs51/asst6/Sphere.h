#ifndef _SPHERE_
#define _SPHERE_

#include "Ray.h"
#include "Shape.h"
#include "Texture.h"
#include "Vec3.h"

class Sphere : public Shape
{
 private:
    Vec3 cen, norm;
    double rad;
    
 public:
    Sphere();
    Sphere(const Vec3& center, double radius, const Texture& texture);
    virtual double GetIntersect(const Ray& ray);
    virtual const Vec3& GetNormal(const Vec3& pos);
};

#endif //_SPHERE_
