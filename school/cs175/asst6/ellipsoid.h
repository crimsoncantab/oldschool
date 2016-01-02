#pragma once
#include <string>
#include "shape.h"
#include <string>
#include <GL/glut.h>
#include "matrix4.h"
#include "shaderstate.h"
#include "rbt.h"
#include <cstddef>

class Ellipsoid :
	public Shape
{
private:

public:

	Ellipsoid(string name, Rbt frame, Vector4 color, double radius, Matrix4 scale) : Shape(name, frame, color),radius_(radius){scale_ = scale;}
	Ellipsoid(Rbt frame, Vector4 color, double radius, Matrix4 scale) : Shape("an ellipsoid", frame, color),radius_(radius){scale_ = scale;}
	Ellipsoid(const Ellipsoid & e) : Shape(e), radius_(e.radius_) {}
private:
	const double radius_;
	void drawShape(ShaderState& glAccess)  {
		glutSolidSphere(radius_, 10, 10);
	}
protected:
	virtual Object3D* clone() { return new Ellipsoid(*this);}

};
class Sphere :
	public Ellipsoid
{
private:

public:

	Sphere(string name, Rbt frame, Vector4 color,  double radius) : Ellipsoid(name, frame, color, radius, Matrix4()){}
	Sphere(Rbt frame, Vector4 color, double radius) : Ellipsoid("a sphere", frame, color, radius, Matrix4()){}
	Sphere(const Sphere & e) : Ellipsoid(e) {}
protected:
	virtual Object3D* clone() { return new Sphere(*this);}
};
