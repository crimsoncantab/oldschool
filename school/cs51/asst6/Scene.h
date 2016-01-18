#ifndef _SCENE_
#define _SCENE_

#include <limits>
#include <vector>

#include "Camera.h"
#include "Image.h"
#include "Light.h"
#include "Ray.h"
#include "Vec3.h"

using namespace std;

class Shape;

class Scene
{
 private:
    vector<Light> lights;
    vector<Shape *> shapes;
    Camera camera;

    vector<Ray> CreateRays(int width, int height);
    Color GetColor(Shape *shape, const Ray& incoming, double lambda);
    
    bool IsShaded(const Vec3& point, const Light& light, Shape *self);
    double MapDimensions(int value, int full);
    
 public:
    Scene();
    ~Scene();

    void Render(int width, int height, string output);
    void AddLight(Light l);
    void AddShape(Shape* s);
    void SetCamera(Camera c);
};

#endif //_SCENE_
