#pragma once

#include <string>
#include "shape.h"
#include <string>
#include <GL/glut.h>
#include "matrix4.h"
#include "shaderstate.h"
#include "rbt.h"
#include <cstddef>
using namespace std;
class RectPrism :
	public Shape
{
public:
	RectPrism(Rbt frame, Vector4 color, double size, Matrix4 scale): Shape("a rectangular prism",frame,color), size_(size){ scale_=scale;}
	RectPrism(string name, Rbt frame, Vector4 color, double size, Matrix4 scale): Shape(name,frame,color), size_(size){scale_ = scale;}
	RectPrism(const RectPrism & r) : Shape(r), size_(r.size_) {}
private:
	const double size_;
	void drawShape(ShaderState& glAccess)  {
		glutSolidCube(size_);
	}
protected:
	virtual Object3D* clone() { return new RectPrism(*this);}
};
class Cube :
	public RectPrism
{
	
public:
	Cube(Rbt frame, Vector4 color, double size): RectPrism("a cube",frame,color,size, Matrix4()){}
	Cube(string name, Rbt frame, Vector4 color, double size): RectPrism(name,frame,color,size, Matrix4()){}
	Cube(const Cube & c) : RectPrism(c) {}
protected:
	virtual Object3D* clone() { return new Cube(*this);}

};
