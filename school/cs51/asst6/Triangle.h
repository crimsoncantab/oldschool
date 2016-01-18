#ifndef _TRIANGLE_
#define _TRIANGLE_

#include <cmath>

// We need to define the value of PI
#ifndef M_PI
#define M_PI 3.1415926535897932385
#endif

#include "Plane.h"
#include "Ray.h"
#include "Shape.h"
#include "Vec3.h"

using namespace std;

class Triangle : public Shape
{
 private:
    Vec3 v1, v2, v3, center;
    Plane plane;
    double furthest;
    
 public:
    Triangle();
    Triangle(const Vec3& vec1, const Vec3& vec2, const Vec3& vec3, const Texture& texture);
    virtual double GetIntersect(const Ray& ray);
    virtual const Vec3& GetNormal(const Vec3& pos);
};

#endif //_TRIANGLE_
