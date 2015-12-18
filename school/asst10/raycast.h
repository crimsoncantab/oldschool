#ifndef RAYCAST_H
#define RAYCAST_H


#include <cassert>
#include "surface.h"


static Vector3 rayTrace(const std::vector <Light>& light, const std::vector <Surface *>& scene, const Ray& ray, const int level);


struct Intersection
{
    double lambda_;																	// lambda == -1 means "no intersection was found"
    int objectId_;
};

static Intersection rayCast(const std::vector <Surface *>& scene, const Ray& ray)	// this is the function that simply goes through all the geometry and 
{																					// finds the closest intersection in the positive side of the ray
    Intersection in;
    in.lambda_ = -1;
    for (std::size_t i = 0; i < scene.size(); ++i)
    {
         const double lambda = scene[i]->intersect(ray);

         if (lambda > 0 && (in.lambda_ < 0 || lambda < in.lambda_))
         {
             in.lambda_ = lambda;
             in.objectId_ = i;
         }
    }
    return in;
}



static Vector3 shadeLight(const Light& light, const std::vector <Surface *>& scene, const Surface& surface, const Vector3& viewDirection, const Vector3& point, const Vector3& normal, const int level)
{
    Vector3 newDirection = light.position_ - point;
	Ray toLight(point + newDirection*1e-5, newDirection);
	Intersection blocksLight = rayCast(scene, toLight);
	if (blocksLight.lambda_ > 0. && blocksLight.lambda_ < 1.-1e-5) return Vector3(0);
    
    const Vector3& lightDirection = (light.position_ - point).normalize();
    const Vector3& lightIntensity = light.intensity_;
    const Vector3 bounceDirection = getBounce(lightDirection, normal);
    
    const Vector3& kd = surface.getDiffuseCoef();
    const Vector3& ks = surface.getSpecularCoef();
    const int n = surface.getExponentSpecular();
    
    const Vector3 diffuse = (kd * lightIntensity) * std::max(0., Vector3::dot(normal, lightDirection));
    const Vector3 specular = (ks * lightIntensity) * std::pow(std::max(0., Vector3::dot(bounceDirection, viewDirection)), n);
    
    return diffuse + specular;
}




static Vector3 shade(const std::vector <Light>& light, const std::vector <Surface *>& scene, const int objectId, const Ray& ray, const Intersection& in, const int level)
{
	if (level > 5) return Vector3(0);
    const Surface& surface = *scene[objectId];
    const Vector3 point = ray.point_ + ray.direction_ * in.lambda_;				// the intersection point
    const Vector3 normal = surface.computeNormal(point);							// the normal at the intersection point
    const Vector3 viewDirection = (ray.point_ - point).normalize();				// the view direction goes from the point to the beginning of the ray (normalized)
    const Vector3& ka = surface.getAmbientCoef();

    assert(std::abs(Vector3::dot(normal, normal) - 1) < 1e-6);						// we check that the normal is pre-normalized

    Vector3 outputColor(0,0,0);
    for (std::size_t i = 0; i < light.size(); ++i)
    {
        outputColor += ka * light[i].intensity_;
        if (Vector3::dot(light[i].position_ - point, normal) > 0)
        {
            outputColor += shadeLight(light[i], scene, surface, viewDirection, point, normal, level);
        }
    }
	if (surface.getMirrorCoef() > 0) {
		Vector3 reflectDirection = getBounce(viewDirection,normal);
		Ray reflected(point+reflectDirection*1e-2, reflectDirection);
		Vector3 reflectedColor = rayTrace(light, scene, reflected, level +1);
		outputColor = (reflectedColor * surface.getMirrorCoef()) + (outputColor * (1-surface.getMirrorCoef()));
	}
    return outputColor;
	//return normal;
}




static Vector3 rayTrace(const std::vector <Light>& light, const std::vector <Surface *>& scene, const Ray& ray, const int level = 0)
{
    const Intersection in = rayCast(scene, ray);								// compute the intersection point 

    if (in.lambda_ < 0) return Vector3(0,0,0);										// if no intersection => return black (background color)

    return shade(light, scene, in.objectId_, ray, in, level);					// ..otherwise compute shade and return that color
}



#endif
