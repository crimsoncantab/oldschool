#include "Cube.h"

Cube::Cube()
{
}

/* 0----------1
   |\         |\
   | \        | \
   |  \       |  \
   |   3----------2
   |   |      |   |
   4---|------5   |
    \  |       \  |
     \ |        \ |
      \|         \|
       7----------6 */

// TODO : Create proper constructor.
//
// Our test image assumes your constructor takes
// a vector of 8 corner points passed in the order
// represented above.

Cube::Cube(const vector<Vec3>& points,const Texture& texture)
    :Shape(texture)
{
    //create 12 triangles, 2 for each face of cube
    vector<Shape*> triangles;
    triangles.push_back(new Triangle(points[0],points[1],points[2],texture));
    triangles.push_back(new Triangle(points[0],points[2],points[3],texture));
    triangles.push_back(new Triangle(points[0],points[3],points[4],texture));
    triangles.push_back(new Triangle(points[4],points[3],points[7],texture));
    triangles.push_back(new Triangle(points[3],points[2],points[7],texture));
    triangles.push_back(new Triangle(points[6],points[2],points[7],texture));
    triangles.push_back(new Triangle(points[6],points[2],points[5],texture));
    triangles.push_back(new Triangle(points[5],points[1],points[2],texture));
    triangles.push_back(new Triangle(points[0],points[1],points[5],texture));
    triangles.push_back(new Triangle(points[0],points[5],points[4],texture));
    triangles.push_back(new Triangle(points[4],points[5],points[7],texture));
    triangles.push_back(new Triangle(points[6],points[5],points[7],texture));

    //put triangles in group
    group = new Group(triangles);

}
Cube::~Cube()
{
    delete group;
}

double Cube::GetIntersect(const Ray& ray)
{
    return group->GetIntersect(ray);
}

const Vec3& Cube::GetNormal(const Vec3& pos)
{

    return group->GetNormal(pos);
}
