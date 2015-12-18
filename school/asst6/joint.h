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
	static double bezerpScalar(double s1, double s2, double sprev, double spost, double t) {
		double vals[4];
		vals[0] = s1;
		vals[1] = s1 + (1./6.)*(s2 - sprev);
		vals[2] = s2 - (1./6.)*(spost - s1);
		vals[3] = s2;

		for (int i = 3; i > 0; i--) {
			for (int j = 0; j < i; j++) {
				vals[j] = (1. - t) * vals[j] + t * vals[j+1];
			}
		}
		return vals[0];
	}
public:

	Joint(string name, Rbt frame, Vector4 color) : Shape(name,frame, color), staticFrame_(frame) {init();}
	Joint(Rbt frame, Vector4 color) : Shape("a rigid joint",frame, color) {init();}
	Joint(const Joint& j) : Shape(j), staticFrame_(j.staticFrame_), yRot_(j.yRot_), xRot_(j.xRot_) {}
	virtual bool usesDefaultTransformBehavior(){return false;}
	virtual void interpretMouseMotion(const int dx, const int dy) {
		xRot_ += 0.5 * dy;
		yRot_ += 0.5 * dx;
		frame_ = staticFrame_ * Rbt(Quaternion::makeYRotation(yRot_));
		beforeChildren_ = Rbt(Quaternion::makeXRotation(xRot_) * Quaternion::makeYRotation(yRot_));
	}

	bool usesArcball() { return false;}
protected:
	virtual Object3D* clone() { return new Joint(*this);}
	virtual void lerp(Object3D * o1, Object3D * o2, double t) {
		Object3D::lerp(o1, o2, t);
		Joint * j1 = dynamic_cast<Joint *>(o1);
		Joint * j2 = dynamic_cast<Joint *>(o2);
		
		//static frame is, well, static; doesn't need lerping
		yRot_ = j1->yRot_ * (1-t) + j2->yRot_ * t;
		xRot_ = j1->xRot_ * (1-t) + j2->xRot_ * t;
		beforeChildren_ = Rbt(Quaternion::makeXRotation(xRot_) * Quaternion::makeYRotation(yRot_));

	}
	virtual void bezerp(Object3D * o1, Object3D * o2, Object3D * prev, Object3D * post, double t) {
		Object3D::bezerp(o1, o2, prev, post, t);
		Joint * j1 = dynamic_cast<Joint *>(o1);
		Joint * j2 = dynamic_cast<Joint *>(o2);
		Joint * jprev = dynamic_cast<Joint *>(prev);
		Joint * jpost = dynamic_cast<Joint *>(post);
		xRot_ = bezerpScalar(j1->xRot_, j2->xRot_, jprev->xRot_, jpost->xRot_, t);
		yRot_ = bezerpScalar(j1->yRot_, j2->yRot_, jprev->yRot_, jpost->yRot_, t);
		beforeChildren_ = Rbt(Quaternion::makeXRotation(xRot_) * Quaternion::makeYRotation(yRot_));
	}

	virtual void save(ostream *out) {
		Object3D::save(out);
		*out<<yRot_<<" "<<xRot_<<" ";
	}
	virtual void load(istream * in) {
		Object3D::load(in);
		*in>>yRot_;
		*in>>xRot_;
	}

};
