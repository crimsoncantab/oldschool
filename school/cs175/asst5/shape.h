#pragma once
#include "object3d.h"
#include <string>
#include <cstddef>

using namespace std;

class Shape : public Object3D //an object3d with color (ostensibly visible)
{
protected:
	Vector4 color_;
	Shape(string name, Rbt frame, Vector4 color) : Object3D(name,frame), color_(color){}
	Shape(Rbt frame, Vector4 color) : Object3D("a shape", frame), color_(color){}
	Shape(const Shape& s) : Object3D(s), color_(s.color_) {}
	virtual void draw(ShaderState& glAccess) {
		sendColor(glAccess);
		drawShape(glAccess);
	}
	virtual void drawShape(ShaderState& glAccess) = 0;
	//draws id into red, instead of drawing color
	virtual void sendColor(ShaderState& glAccess) {
		safe_glVertexAttrib4f(glAccess.h_vColor_, color_[0], color_[1], color_[2], color_[3]);
	}
	virtual void drawId(ShaderState& glAccess) {
		safe_glVertexAttrib4f(glAccess.h_vColor_, Object3D::convertIdToColor(id_), 0, 0, 1);
		drawShape(glAccess);
	}
	//virtual void lerp(Object3D * o1, Object3D * o2, double t) {
		//we don't need this right now
		//Shape * s1 = dynamic_cast<Shape *>(o1);
		//Shape * s2 = dynamic_cast<Shape *>(o2);
		//color_ = s1->color_.lerp(s2->color_, t);
	//}
};
