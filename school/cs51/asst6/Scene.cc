#include "Scene.h"
#include "Shape.h"

Scene::Scene()
{
}

Scene::~Scene()
{
    for (unsigned int i = 0; i < shapes.size(); i++)
        delete shapes[i];
}

vector<Ray> Scene::CreateRays(int width, int height)
{
    // Ratios. Prevents stretching in the image.
    double wRatio = width > height ? 1 : (double)width / (double)height;
    double hRatio = height > width ? 1 : (double)height / (double)width;
    
    vector<Ray> rays;
    for(int j = 0; j < height; j++)
    {
        for(int i = 0; i < width; i++)
        {
            // Finds the xoffset and yoffset as doubles
            double xoff = MapDimensions(i, width) * wRatio;
            double yoff = MapDimensions(j, height) * hRatio;
            rays.push_back(camera.Shoot(xoff, yoff));
        }
    }
    
    return rays;
}

Color Scene::GetColor(Shape *shape, const Ray& incoming, double lambda)
{
    // The point where the ray intersects the shape
    Vec3 point = incoming.GetVec(lambda);
    
    // The normal at that point
    Vec3 normal = shape->GetNormal(point);
    
    // Diffuse color
    double diffuseCoeff = 0;
    for(unsigned int i = 0; i < lights.size(); i++)
    {
        Light currLight = lights[i];
        Vec3 toLight = (currLight.GetPos() - point).normalize();
        
        // Determine how much to color based on the angle to the light
        double tmp = abs(dot(toLight, normal));
        if (tmp < 0 || tmp > 1)
        {
            continue;
        }
        tmp *= lights[i].GetIntensity();
        
        // Only include the illumination from this light if it is not
        // blocked by another shape
        if(!IsShaded(point, currLight, shape))
        {
            diffuseCoeff += tmp;
        }
    }
    
    Texture texture = shape->GetTexture(); 
    double ambientCoeff = texture.GetAmbientCoeff();
    Color baseColor = texture.GetBaseColor();
    
    double finalCoeff = ambientCoeff + diffuseCoeff * (1 - ambientCoeff);
    Color diffuseColor = finalCoeff * baseColor;

    // TODO : Combine diffuseColor with the correct output of GetReflectColor
    //        to support reflective surfaces.
    
    return diffuseColor;
}

bool Scene::IsShaded(const Vec3& point, const Light& light, Shape *self)
{
    Ray toLight = Ray(point, light.GetPos() - point);
    double lightDist = dist(point, light.GetPos());
    
    for(unsigned int i = 0; i < shapes.size(); i++)
    {
        if(shapes[i] == self)
        {
            continue;
        }
        
        double lambda = shapes[i]->GetIntersect(toLight);
        
        // Only if the intersection exists and is between the point
        // and the light is the point shaded
        if(lambda > 0 && lambda < lightDist)
        {
            return true;
        }
    }
    
    return false;
}

double Scene::MapDimensions(int value, int full)
{
    return ((double)value + 0.5) / (double)full - 0.5;
}

void Scene::Render(int width, int height, string output)
{
    Image* im = new Image(width, height);
    vector<Ray> rays = CreateRays(width, height);

    //iterates through every ray in image and finds color along that ray
    for (unsigned int i = 0; i < rays.size(); i++)
    {
        //finds the closest shape, if any, along the ith ray
        double closestDist;
        Shape* closest = NULL;
        for (unsigned int j = 0; j < shapes.size(); j++)
        {
            double lambda = shapes[j]->GetIntersect(rays[i]);
            if (lambda < 0)
                continue;
            if (!closest || lambda < closestDist)
            {
                closest = shapes[j];
                closestDist = lambda;
            }
        }
        //sets color to black unless a shape is intersected
        Color c = Color();
        if (closest)
            c = GetColor(closest, rays[i], closestDist);

        im->SetPixel(i, c);
    }
    //creates image file
    im->OutputBitmap(output);

}
//adds a light to the scene
void Scene::AddLight(Light l)
{
    lights.push_back(l);
}

//adds a shape to the scene
void Scene::AddShape(Shape *s)
{
    shapes.push_back(s);
}

//positions the camera
void Scene::SetCamera(Camera c)
{
    camera = c;
}
