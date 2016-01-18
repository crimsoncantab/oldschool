#ifndef _CUBE_
#define _CUBE_

#include <vector>

#include "Ray.h"
#include "Shape.h"
#include "Texture.h"
#include "Triangle.h"
#include "Vec3.h"
#include "Group.h"

class Cube : public Shape
{
private:
    Group * group;
public:
    Cube();
    ~Cube();
    Cube(const vector<Vec3>& points,const Texture& texture);
    virtual double GetIntersect(const Ray& ray);
    virtual const Vec3& GetNormal(const Vec3& pos);
};

#endif //_CUBE_
