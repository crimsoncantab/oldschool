#ifndef SURFACE_H
#define SURFACE_H


#include "raylight.h"
#include <cmath>



static Vector3 getBounce(const Vector3& v, const Vector3& n)
{
	return (n * 2 * Vector3::dot(n,v))- v;
}



class Surface
{
    Vector3 ka_;
    Vector3 kd_;
    Vector3 ks_;
    int exponentSpecular_; 
	double mirrorCoef_;
    
public:
    Surface()
    : ka_(0.1,0.1,0.1), kd_(0.8,0.8,0.8), ks_(0,0,0), exponentSpecular_(32), mirrorCoef_(0)
    {}
    
    Surface(const Vector3& ka, const Vector3& kd, const Vector3& ks, const int exponentSpecular, const double mirrorCoef)
    : ka_(ka), kd_(kd), ks_(ks), exponentSpecular_(exponentSpecular), mirrorCoef_(mirrorCoef)
    {}

    virtual ~Surface() {}

    virtual double intersect(const Ray& ray) const = 0;
    virtual Vector3 computeNormal(const Vector3& p) const = 0;				// returns  a *normalized* normal vector
        
    const Vector3& getAmbientCoef() const
    {
        return ka_;
    }
    const Vector3& getDiffuseCoef() const
    {
        return kd_;
    }
    const Vector3& getSpecularCoef() const
    {
        return ks_;
    }
    int getExponentSpecular() const
    {
        return exponentSpecular_;
    }
	double getMirrorCoef() const
	{
		return mirrorCoef_;
	}
};



class Sphere : public Surface
{
    Vector3 center_;
    double radius_;

public:
    Sphere(const Vector3& center, const double radius,
           const Vector3& ka, const Vector3& kd, const Vector3& ks, const int exponentSpecular, const double mirrorCoef)
    : Surface(ka, kd, ks, exponentSpecular, mirrorCoef), center_(center), radius_(radius)
    {}

    Sphere(const Vector3& center, const double radius)
    : Surface(), center_(center), radius_(radius)
    {}
    
    double intersect(const Ray& ray) const
    {
		double a = Vector3::dot(ray.direction_, ray.direction_),
			b = 2 * (Vector3::dot(ray.direction_, ray.point_ - center_)),
		c = Vector3::dot(center_, center_) - radius_ * radius_ +
			Vector3::dot(ray.point_, ray.point_) - 2 * Vector3::dot(ray.point_, center_);
		double radical = b * b - 4. * a * c;
		if (radical < 0) return -1;
		double radsqrt = std::sqrt(radical);
		double lamda1 = (-b + radsqrt) / (2 * a);
		double lamda2 = (-b - radsqrt) / (2 * a);

		if (lamda1 < 0) {
			if (lamda2 < 0) return -1;
			else return lamda2;
		}
		else {
			if (lamda2 < 0) return lamda1;
			else return std::min(lamda1, lamda2);
		}
    }
    Vector3 computeNormal(const Vector3& p) const
    {
		return (p-center_).normalize();
    }
};


class Triangle : public Surface
{
    Vector3 point_[3];
	Vector3 normal_;

public:
    Triangle(const Vector3& p0, const Vector3& p1, const Vector3& p2,
             const Vector3& ka, const Vector3& kd, const Vector3& ks, const int exponentSpecular, const double mirrorCoef)
    : Surface(ka, kd, ks, exponentSpecular, mirrorCoef)
    {
        point_[0] = p0;
        point_[1] = p1;
        point_[2] = p2;
		normal_ = (Vector3::cross(point_[2]-point_[1], point_[0] - point_[1])).normalize();
    }
    
    Triangle(const Vector3& p0, const Vector3& p1, const Vector3& p2)
    : Surface()
    {
        point_[0] = p0;
        point_[1] = p1;
        point_[2] = p2;
    }
    
    double intersect(const Ray& ray) const
    {
		double d = Vector3::dot(point_[0] * -1, normal_);
		double denom = Vector3::dot(normal_, ray.direction_);
		if (denom == 0) return -1;
		double lamda= (-d - Vector3::dot(normal_, ray.point_))/denom;
		Vector3 intersect = ray.point_ + ray.direction_ * lamda;

		Vector3 v0i = intersect-point_[0], v1i=intersect-point_[1], v2i = intersect-point_[2],
			v01 = point_[1]-point_[0], v12 = point_[2]-point_[1], v20 = point_[0]-point_[2];
		bool v0dotnpos = (Vector3::dot(Vector3::cross(v01, v0i),normal_) > 0);
		bool v1dotnpos = (Vector3::dot(Vector3::cross(v12, v1i),normal_) > 0);
		bool v2dotnpos = (Vector3::dot(Vector3::cross(v20, v2i),normal_) > 0);
		if ((v0dotnpos && v1dotnpos && v2dotnpos) || (!v0dotnpos && !v1dotnpos && !v2dotnpos)){
			return lamda;
		}
		return -1;
	}
    Vector3 computeNormal(const Vector3& p) const
    {
		return normal_;
    }
};


class Plane : public Surface
{
    Vector3 point_;
    Vector3 normal_;

public:
    Plane(const Vector3& point, const Vector3& normal, 
          const Vector3& ka, const Vector3& kd, const Vector3& ks, const int exponentSpecular, const double mirrorCoef)
    : Surface(ka, kd, ks, exponentSpecular, mirrorCoef), point_(point), normal_(normal)
    {}
    
    Plane(const Vector3& point, const Vector3& normal)
    : point_(point), normal_(normal)
    {}
    
    double intersect(const Ray& ray) const
    {
		double d = Vector3::dot(point_ * -1, normal_);
		double denom = Vector3::dot(normal_, ray.direction_);
		if (denom == 0) return -1;
		return (-d - Vector3::dot(normal_, ray.point_))/denom;
    }
    Vector3 computeNormal(const Vector3& p) const
    {
		return Vector3(normal_ ).normalize();
    }
};

#endif
