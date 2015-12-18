#pragma once

#include <string>
#include "shape.h"
using namespace std;
class Cube :
	public Shape
{
	
public:
	Cube(Rbt frame, Vector4 color, double size): Shape("a cube",frame,color), size_(size){}
	Cube(string name, Rbt frame, Vector4 color, double size): Shape(name,frame,color), size_(size){}
private:
	const double size_;
	void draw(ShaderState& glAccess)  {
		safe_glVertexAttrib4f(glAccess.h_vColor_, color_[0], color_[1], color_[2], color_[3]);
		glutSolidCube(size_);
	}

};
