#include <iostream>

#include "Camera.h"
#include "Circle.h"
#include "Color.h"
#include "Cube.h"
#include "Group.h"
#include "Image.h"
#include "Light.h"
#include "Plane.h"
#include "Ray.h"
#include "Scene.h"
#include "Shape.h"
#include "Sphere.h"
#include "Texture.h"
#include "Triangle.h"
#include "Vec3.h"

using namespace std;

int main()
{

    Scene* myScene = new Scene();
/*
    myScene->SetCamera(Camera(Vec3(-5, 5, -10), Vec3()));
    myScene->AddLight(Light(Vec3(10, 20, -20), 1));
    myScene->AddShape(new Plane(Vec3(0, -1, 0), Vec3(0, 1, 0), Texture(Color(0, 0.5, 0), 0)));
    myScene->AddShape(new Circle(Vec3(0, 2, 5), Vec3(0, 0, -1), 3.5, Texture(Color(1, 0, 0), 0)));
    myScene->AddShape(new Sphere(Vec3(3.5, 0, 0), 1.2, Texture(Color(0, 0, 1), 0)));
    vector<Vec3> cubeVecs;
    cubeVecs.push_back(Vec3(-1, 1, 1));
    cubeVecs.push_back(Vec3( 1, 1, 1));
    cubeVecs.push_back(Vec3( 1, 1, -1));
    cubeVecs.push_back(Vec3(-1, 1, -1));
    cubeVecs.push_back(Vec3(-1, -1, 1));
    cubeVecs.push_back(Vec3( 1, -1, 1));
    cubeVecs.push_back(Vec3( 1, -1, -1));
    cubeVecs.push_back(Vec3(-1, -1, -1));
    myScene->AddShape(new Cube(cubeVecs, Texture(Color(0, 1, 1), 0)));
    myScene->Render(800,600,"part2-test1.bmp");
*/
/*
    myScene->SetCamera(Camera(Vec3(-5, 5, -10), Vec3()));
    myScene->AddLight(Light(Vec3(10, 20, -20), 1));
    vector<Shape *> shapes;
    shapes.push_back(new Plane(Vec3(0, -1, 0), Vec3(0, 1, 0), Texture(Color(0, 0.5, 0), 0)));
    shapes.push_back(new Circle(Vec3(0, 2, 5), Vec3(0, 0, -1), 3.5, Texture(Color(1, 0, 0), 0)));
    shapes.push_back(new Sphere(Vec3(3.5, 0, 0), 1.2, Texture(Color(0, 0, 1), 0)));
    vector<Vec3> cubeVecs;
    cubeVecs.push_back(Vec3(-1, 1, 1));
    cubeVecs.push_back(Vec3( 1, 1, 1));
    cubeVecs.push_back(Vec3( 1, 1, -1));
    cubeVecs.push_back(Vec3(-1, 1, -1));
    cubeVecs.push_back(Vec3(-1, -1, 1));
    cubeVecs.push_back(Vec3( 1, -1, 1));
    cubeVecs.push_back(Vec3( 1, -1, -1));
    cubeVecs.push_back(Vec3(-1, -1, -1));
    shapes.push_back(new Cube(cubeVecs, Texture(Color(0, 1, 1), 0)));
    myScene->AddShape(new Group(shapes));
    myScene->Render(800,600,"part2-test2.bmp");
*/

/*
    myScene->SetCamera(Camera(Vec3(0, 0, -10),Vec3()));
    myScene->AddLight(Light(Vec3(0, -20, 0), 1));
    myScene->AddShape(new Plane(Vec3(0, 10, 0), Vec3(0, 1, 0), Texture(Color(0.5, 0.5, 0), 0)));
    myScene->AddShape(new Circle(Vec3(0,5,5), Vec3(1, 1, -1), 2, Texture(Color(0,1,0),0)));
    myScene->AddShape(new Sphere(Vec3(-5,0,0), .5, Texture(Color(.5,.25,.5),0)));
    vector<Vec3> cubeVecs;
    cubeVecs.push_back(Vec3(0, 0, 1));
    cubeVecs.push_back(Vec3(0, 1, 0));
    cubeVecs.push_back(Vec3(0, 0, -1));
    cubeVecs.push_back(Vec3(0, -1, 0));
    cubeVecs.push_back(Vec3(1, 0, 1));
    cubeVecs.push_back(Vec3( 1, 1, 0));
    cubeVecs.push_back(Vec3( 1, 0, -1));
    cubeVecs.push_back(Vec3(1, -1, 0));
    myScene->AddShape(new Cube(cubeVecs, Texture(Color(.75, 1, 0), 0)));
    myScene->Render(800,600,"part2-test3.bmp");
*/

    myScene->SetCamera(Camera(Vec3(0,0,-10),Vec3()));
    myScene->AddLight(Light(Vec3(-10,0,0), 1));
    myScene->AddLight(Light(Vec3(-10,0,-5), 1));
    for (int i = 1; i < 11; i++)
        myScene->AddShape(new Sphere(Vec3(i-6,0,0), i/2, Texture(Color(.1 * i, 0, .1 * (10 - i)), 0)));
    myScene->Render(800,600,"contest.bmp");
    

/*
    myScene->SetCamera(Camera(Vec3(0, 0, -10),Vec3()));
    myScene->AddLight(Light(Vec3(0, 0, -5), 1));
    myScene->AddShape(new Plane(Vec3(0, 10, 0), Vec3(0, 1, 0), Texture(Color(0.5, 0.5, 0), 0)));
    myScene->AddShape(new Plane(Vec3(0, -10, 0), Vec3(0, 1, 0), Texture(Color(0, 0, 1), 0)));
    myScene->AddShape(new Triangle(Vec3(-5, 5, 10), Vec3(-5, -5, 10), Vec3(5, -5, 10), Texture(Color(0.5, 0.5, 0.5), 0)));
    myScene->Render(800,600,"part1-test2.bmp");
*/

    return 0;
}
