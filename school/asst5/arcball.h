#ifndef ARCBALL_H
#define ARCBALL_H
#include "object3d.h"
#include "shape.h"
#include <string>
#include "shaderstate.h"
#include "quaternion.h"
#include "rbt.h"
#include <GL/glut.h>
#include <cstddef>




class Arcball :
	public Shape
{
private:
	Object3D* toRotate_;
	double radius_;
	Vector3 lastPoint_;
	Vector2 centerScreen_;
	double radiusScreen_;
	Rbt * eyeFrame_;

	static bool getScreenSpaceCircle(const Vector3& center, double radius,						// camera/eye coordinate info for sphere
                                 const Matrix4& projection,									// projection matrix
								 const double frust_near, const double frust_fovy,			// if used asst2/starter or asst2/solution, these are declared at the top of HelloWorld3D.cpp
                                 const int screen_width, const int screen_height, 			// viewport size
                                 Vector2 *out_center, double *out_radius)					// output data in screen coordinates
	{																							// returns false if the arcball is behind the viewer
		// get post projection canonical coordinates
		Vector3 postproj = projection * center;
		double w = projection[3][0] * center[0] + projection[3][1] 
		  * center[1] + projection[3][2] * center[2] + projection[3][3] * 1.0;
		double winv = 0.0;
		if (w != 0.0) {
		  winv = 1.0 / w;
		}
		postproj *= winv;

		// convert to screen pixel space
		(*out_center)[0] = postproj[0] * (double)screen_width/2.0 + ((double)screen_width-1.0)/2.0;
		(*out_center)[1] = postproj[1] * (double)screen_height/2.0 + ((double)screen_height-1.0)/2.0;

		// determine some overall radius
		double dist = center[2];
		if (dist < frust_near) {
			*out_radius = -radius/(dist * tan(frust_fovy * CS175_PI/360.0));
		}
		else {
			*out_radius = 1.0;
			return false;
		}

		*out_radius *= screen_height * 0.5;
		return true;
	}

	virtual void drawShape(ShaderState& glAccess) {
		glutWireSphere(radius_, 20, 20);
	}
	
	Vector3 getCenter(Rbt eyeFrame) {
		Rbt myFrame;
		
		if(toRotate_->usingWorldSky()) {
			myFrame = Rbt();
		} else {
			myFrame = toRotate_->getFrame();
		}
		return (eyeFrame.getInverse() * myFrame).getTranslation();
		
	}

	Vector3 calcClickedPoint(int mouseX, int mouseY) {
		//find the z coord
		Vector2 fromCenter = Vector2(mouseX, mouseY) - centerScreen_;

		//if outside circle, z coord is 0
		double z= 0;
		double len = fromCenter.length();
		if (len < radiusScreen_) {
			//find z coord if inside circle
			z=std::sqrt((radiusScreen_ * radiusScreen_) - (len * len));
		}
		return Vector3(fromCenter[0], fromCenter[1], z); 
	}
protected:
	virtual Object3D* clone() { return new Arcball(*this);}
public:

	Arcball(string name, Vector4 color, Object3D* toRotate, double radius) : Shape(name, (toRotate->usingWorldSky()) ? Rbt() : toRotate->getFrame(), color), radius_(radius), toRotate_(toRotate){}
	Arcball(Vector4 color, Object3D* toRotate, double radius) : Shape("arcball", (toRotate->usingWorldSky()) ? Rbt() : toRotate->getFrame(), color), radius_(radius), toRotate_(toRotate){}
	Arcball(const Arcball & a) : Shape(a), radius_(a.radius_), toRotate_(NULL) {}
	virtual void draw(ShaderState& glAccess, Rbt& eyePoseInverse) {
		if (toRotate_->usesArcball()) {
			Rbt MVM = eyePoseInverse * (toRotate_->usingWorldSky() ? frame_ : toRotate_->getFrame());
			sendModelViewNormalMatrix(MVM.getMatrix(), glAccess);
			Shape::draw(glAccess);
		}
	}

	void setObjectToRotate(Object3D* toRotate){
		toRotate_ = toRotate;
		if(toRotate_->usingWorldSky()) {
		frame_ = Rbt();
		}
	}
	virtual void drawId(ShaderState& glAccess, Rbt& eyePoseInverse) {}

	bool setScreenSpaceCircle(Rbt& eyeFrame, const Matrix4& projection,									
								 const double frust_near, const double frust_fovy,			
                                 const int screen_width, const int screen_height) {
		eyeFrame_ = &eyeFrame;
		return getScreenSpaceCircle(getCenter(eyeFrame), radius_,projection,
			frust_near,frust_fovy,screen_width, screen_height, &centerScreen_, &radiusScreen_);
	}


	void startRotation(int mouseX, int mouseY) {
		lastPoint_ = calcClickedPoint(mouseX, mouseY).normalize();
	}

	Rbt updateRotation(int mouseX, int mouseY) {
		Vector3 nextPoint = calcClickedPoint(mouseX, mouseY).normalize();
		double dot = Vector3::dot(lastPoint_, nextPoint);
		Vector3 cross;
		if (toRotate_->usingWorldSky()){
			cross = Vector3::cross(nextPoint, lastPoint_);
		}
		else {
			cross = Vector3::cross(lastPoint_, nextPoint);
		}
		Quaternion rotation = Quaternion(dot, cross);
		Rbt transform = Rbt(rotation);
		lastPoint_ = nextPoint;
		return transform;
	}
	virtual bool usesArcball() { return false; }
	virtual bool isEditable() {return false; }
	virtual void lerp(Object3D * o1, Object3D * o2, double t) {
		//don't need to lerp arcballs
	}
};
#endif

