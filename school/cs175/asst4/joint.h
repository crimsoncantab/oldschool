#pragma once
#include "object3d.h"
#include <GL/glut.h>
#include "matrix4.h"
#include "shaderstate.h"
#include "shape.h"
#include "rbt.h"
#include "ellipsoid.h"
#include <vector>
#include <cstddef>

using namespace std;
class Joint :
	public Shape
{
private:
	Rbt staticFrame_;
	double yRot_;
	double xRot_;
	void init() {
		xRot_ = 0;
		yRot_ = 0;
	}
	void draw(ShaderState &glAccess) {}
	void drawShape(ShaderState &glAccess) {}
	void sendColor(ShaderState &glAccess) {}
public:

	Joint(string name, Rbt frame, Vector4 color) : Shape(name,frame, color), staticFrame_(frame) {init();}
	Joint(Rbt frame, Vector4 color) : Shape("a rigid joint",frame, color) {init();}
	virtual bool usesDefaultTransformBehavior(){return false;}
	virtual void interpretMouseMotion(const int dx, const int dy) {
		xRot_ += 0.5 * dy;
		yRot_ += 0.5 * dx;
		frame_ = staticFrame_ * Rbt(Quaternion::makeYRotation(yRot_));
		beforeChildren_ = Rbt(Quaternion::makeXRotation(xRot_) * Quaternion::makeYRotation(yRot_));
		/*for (std::size_t i = 0; i < children_.size(); ++i)
		{
			children_[i]->preApplyTransformation(Rbt(Quaternion::makeXRotation(xRot_) * Quaternion::makeYRotation(yRot_)));
		}*/
	}

	bool usesArcball() { return false;}

};
