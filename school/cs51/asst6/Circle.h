#ifndef _CIRCLE_
#define _CIRCLE_

#include "Plane.h"
#include "Ray.h"
#include "Shape.h"
#include "Texture.h"
#include "Vec3.h"

class Circle : public Shape
{
 private:
    Vec3 center;
    Plane plane;
    double radius;

 public:
    Circle();
    Circle(const Vec3& c, const Vec3& n, const double r, const Texture& texture);
    virtual double GetIntersect(const Ray& ray);
    virtual const Vec3& GetNormal(const Vec3& pos);
};

#endif //_CIRCLE_
