#pragma once
#include "shape.h"
#include <string>
#include <GL/glut.h>
#include "matrix4.h"
#include "shaderstate.h"
#include "rbt.h"
#include "cube.h"
#include "ellipsoid.h"
#include "joint.h"
#include <cstddef>

class Robot :
	public Cube
{
public:

	Robot(string name, Rbt frame, Vector4 color) : Cube(name,frame,color,1.0){createAppendages();}
	Robot(Rbt frame, Vector4 color) : Cube("a robot",frame,color,1.0){createAppendages();}
	Robot(const Robot& r) : Cube(r) {}
	bool hasCamera() {return true;}

private:
	Ellipsoid* makeAppendage() { return new Ellipsoid(Rbt(Vector3(0, -.3, 0)), color_, .5, Matrix4::makeScale(Vector3(.5, 1, .5)));}
	void createAppendages() {
		Rbt rShFrame = Rbt(Vector3(.7, .5, 0));
		Joint* rShoulder = new Joint(name_ + ": Right Shoulder", rShFrame, color_);
		this->addChild(rShoulder);
		
		Ellipsoid* urArm = makeAppendage();
		urArm->name_ = (name_ + ": Upper Right Arm");
		urArm->setId(rShoulder->getId());
		rShoulder->addChild(urArm);
		
		Rbt rElbFrame = Rbt(Vector3(0, -.8, 0));
		Joint* rElbow = new Joint(name_ + ": Right Elbow", rElbFrame, color_);
		rShoulder->addChild(rElbow);
		
		Ellipsoid* rForearm = makeAppendage();
		rForearm->name_ = (name_ + ": Right Forearm");
		rForearm->setId(rElbow->getId());
		rElbow->addChild(rForearm);
		
		Rbt lShFrame = Rbt(Vector3(-.7, .5, 0));
		Joint* lShoulder = new Joint(name_ + ": Left Shoulder", lShFrame, color_);
		this->addChild(lShoulder);
		
		Ellipsoid* ulArm = makeAppendage();
		ulArm->setId(lShoulder->getId());
		ulArm->name_ = (name_ + ": Upper Left Arm");
		lShoulder->addChild(ulArm);
		
		Joint* lElbow = new Joint(name_ + ": Left Elbow", rElbFrame, color_);
		lShoulder->addChild(lElbow);
		
		Ellipsoid* lForearm = makeAppendage();
		lForearm->setId(lElbow->getId());
		lForearm->name_ = (name_ + ": Left Forearm");
		lElbow->addChild(lForearm);

		
		Rbt rHipFrame = Rbt(Vector3(.3, -.4, 0));
		Joint* rHip = new Joint(name_ + ": Right Hip", rHipFrame, color_);
		this->addChild(rHip);
		

		Ellipsoid* urLeg = makeAppendage();
		urLeg->setId(rHip->getId());
		urLeg->name_ = (name_ + ": Upper Right Leg");
		rHip->addChild(urLeg);

		Joint* rKnee = new Joint(name_ + ": Right Knee", rElbFrame, color_);
		rHip->addChild(rKnee);
		
		Ellipsoid* rlLeg = makeAppendage();
		rlLeg->setId(rKnee->getId());
		rlLeg->name_ = (name_ + ": Lower Right Leg");
		rKnee->addChild(rlLeg);

				
		Rbt lHipFrame = Rbt(Vector3(-.3, -.4, 0));
		Joint* lHip = new Joint(name_ + ": Left Hip", lHipFrame, color_);
		this->addChild(lHip);
		
		Ellipsoid* ulLeg = makeAppendage();
		ulLeg->name_ = (name_ + ": Upper Left Leg");
		ulLeg->setId(lHip->getId());
		lHip->addChild(ulLeg);

		Joint* lKnee = new Joint(name_ + ": Left Knee", rElbFrame, color_);
		lHip->addChild(lKnee);
		
		Ellipsoid* llLeg = makeAppendage();
		llLeg->name_ = (name_ + ": Lower Left Leg");
		llLeg->setId(lKnee->getId());
		lKnee->addChild(llLeg);

		Rbt neckFrame = Rbt(Vector3(0, .5, 0));
		Joint* neck = new Joint(name_ + ": Neck", neckFrame, color_);
		this->addChild(neck);
		
		Rbt neckCircFrame = Rbt(Vector3(0, .1, 0));
		Sphere* neckCirc = new Sphere(name_ + ": Neck", neckCircFrame, color_, .2);
		neckCirc->setId(neck->getId());
		neck->addChild(neckCirc);

		Rbt headFrame = Rbt(Vector3(0, .65, 0));
		Sphere* head = new Sphere(name_ + ": Head", headFrame, color_, .6);
		head->setId(neck->getId());
		neck->addChild(head);
		
	}

protected:
	virtual Object3D* clone() { return new Robot(*this);}
};
