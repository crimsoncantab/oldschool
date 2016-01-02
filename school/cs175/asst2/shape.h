#pragma once
#include "object3d.h"
#include <string>

using namespace std;

class Shape : public Object3D //an object3d with color (ostensibly visible)
{
protected:
	Vector3 color_;
	Shape(string name, Matrix4 frame, Vector3 color) : Object3D(name,frame), color_(color){}
	Shape(Matrix4 frame, Vector3 color) : Object3D("a shape", frame), color_(color){}
	virtual void draw(ShaderState& glAccess) = 0;
};
