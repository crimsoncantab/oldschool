#ifndef _GROUP_
#define _GROUP_

#include <vector>

#include "Ray.h"
#include "Shape.h"
#include "Vec3.h"

class Group : public Shape
{
 private:
    vector<Shape *> shapes;
    Shape* intersected;
    
 public:
    Group();
    ~Group();
    Group(const vector<Shape *> s);
    virtual double GetIntersect(const Ray& ray);
    virtual const Vec3& GetNormal(const Vec3& pos);
};

#endif //_GROUP_
