#include "Group.h"

Group::Group()
{
}

Group::Group(const vector<Shape *> s)
    : shapes(s)
{
    intersected = NULL;
}

Group::~Group()
{
    for (unsigned int i = 0; i < shapes.size(); i++)
        delete shapes[i];
}

double Group::GetIntersect(const Ray& ray)
{
    intersected = NULL;
    double closest = -1, lambda;
    for (unsigned int i = 0; i < shapes.size(); i++)
    {
        lambda = shapes[i]->GetIntersect(ray);

        if (lambda < 0)
            continue;

        if (!intersected || lambda < closest)
        {
            closest = lambda;
            intersected = shapes[i];
        }
    }
    if (intersected)
        texture = intersected->GetTexture();
    return closest;
}

const Vec3& Group::GetNormal(const Vec3& pos)
{

    return intersected->GetNormal(pos);
}
