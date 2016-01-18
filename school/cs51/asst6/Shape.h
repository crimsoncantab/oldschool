#ifndef _SHAPE_
#define _SHAPE_

#include "Color.h"
#include "Ray.h"
#include "Scene.h"
#include "Texture.h"

using namespace std;

class Shape
{
 protected:
    Texture texture;
    
 public:
    Shape();
    Shape(const Texture& texture);
    virtual ~Shape() { }
    
    Texture GetTexture();
    
    virtual double GetIntersect(const Ray& ray) = 0;
    virtual const Vec3& GetNormal(const Vec3& pos) = 0;
};

#endif //_SHAPE_
