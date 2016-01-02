#ifndef RAYLIGHT_H
#define RAYLIGHT_H


#include "vec.h"



struct Light
{
    Vector3 position_;
    Vector3 intensity_;
    
    Light() {}
    Light(const Vector3& position, const Vector3& intensity) : position_(position), intensity_(intensity) {}
};



struct Ray
{
    Vector3 point_;
    Vector3 direction_;
    
    Ray(const Vector3& point, const Vector3& direction) : point_(point), direction_(direction) {}
};




#endif

