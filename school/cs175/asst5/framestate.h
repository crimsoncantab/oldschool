#pragma once
#include <vector>
#include <string>
#include <list>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include "ppm.h"
#include "matrix4.h"
#include "shader.h"
#include "object3d.h"
#include "shape.h"
#include "floor.h"
#include "cube.h"
#include "skycam.h"
#include "shaderstate.h"
#include "rbt.h"
#include "quaternion.h"
#include "arcball.h"
#include "robot.h"
#include "keyframe.h"


using namespace std;
class FrameState :
	public KeyFrame
{
private:
	Arcball* arcball_;
	Object3D* currentMut_;
	Object3D* currentEye_;
	Vector3 light_;
	Rbt auxilaryFrame_;
	int eyePoseIndex_;																	// index of current camera
	int mutableIndex_;
	bool isAnim;
	

	void calcAuxFrame(){
		Rbt objectFrame = currentMut_->getFrame();
		if (currentMut_->usingWorldSky()) {
				//uses axes of skycam, origin of world
				auxilaryFrame_ = Rbt(objectFrame.getRotation());
		} else {
			Rbt eyeFrame = currentEye_->getFrame();
			//axes of current eyePose, origin of object
			auxilaryFrame_ = Rbt(objectFrame.getTranslation(), eyeFrame.getRotation());
		}
	}
	bool doTransformChecks(const double dx, const double dy) {
		if (currentMut_->getId() != currentEye_->getId() && !currentMut_->editableRemotely()) return false;
		if (!currentMut_->usesDefaultTransformBehavior()) {currentMut_->interpretMouseMotion(dx, dy); return false;}
		return true;
	}

	void applyTransform(Rbt transform) {
		calcAuxFrame();
		currentMut_->preApplyTransformation(auxilaryFrame_ * transform * auxilaryFrame_.getInverse());
	}
	int mod(int x, int m) {
		return (x%m + m)%m;
	}
	void printState() {
		cout << "Manipulating " << currentMut_->name_ << "; " << "using " << currentEye_->name_ << " as camera."<< endl;
	}
	void changeMutable(int id, int defIndex) {
		for (size_t i = 0; i < objects_.size(); ++i)
		{
			Object3D* temp = objects_[i];
			Object3D* clicked = temp->findId(id);
			if (clicked != NULL) {changeMutable(clicked); return;}
		}
		//otherwise, skycam
		changeMutable(objects_[defIndex]);
	}
public:

	FrameState(vector <Object3D*> objects, Arcball* arcball, Vector3 light) : KeyFrame(objects), arcball_(arcball), light_(light)
	{
		eyePoseIndex_ = 0;
		mutableIndex_ = 0;
		currentMut_ = objects_[mutableIndex_];
		currentEye_ = objects_[eyePoseIndex_];
		isAnim = false;
	}
	virtual void copyFrom(KeyFrame f) {
		KeyFrame::copyFrom(f);
		changeMutable(currentMut_->getId(),0);
		currentEye_ = objects_[eyePoseIndex_];
		arcball_->setObjectToRotate(currentMut_);
	}
	void changeMutable(int indexDelta) {
		int mutLen = objects_.size();
		do {
			mutableIndex_ += indexDelta;
			mutableIndex_ = mod(mutableIndex_, mutLen);
			currentMut_ = objects_[mutableIndex_];
		} while(!currentMut_->isEditable());
		arcball_->setObjectToRotate(currentMut_);
		printState();
	}
	void changeMutable(Object3D* newMut) {
		if (!newMut->isEditable()) return;
		currentMut_ = newMut;
		arcball_->setObjectToRotate(currentMut_);
		printState();
	}
	void changeCamera(int indexDelta) {
		int mutLen = objects_.size();
		do {
			eyePoseIndex_ += indexDelta;
			eyePoseIndex_ = mod(eyePoseIndex_, mutLen);
			currentEye_ = objects_[eyePoseIndex_];
		} while(!currentEye_->hasCamera());
		printState();
	}

	void changeMutable(int x, int y, int defIndex) {
		float * point = new float[0];
		glReadPixels(x,y, 1, 1, GL_RED, GL_FLOAT, point);
		int id = Object3D::convertColorToId(point[0]);
		changeMutable(id, defIndex);
	}
	void draw(ShaderState glAccess, bool pickMode) {
		Rbt eyePoseInverse = currentEye_->getFrame().getInverse();
		const Vector3 lightE = eyePoseInverse * light_;													// light direction in eye coordinates
		safe_glUniform3f(glAccess.h_uLightE_, lightE[0], lightE[1], lightE[2]);
		// draw objects
		for (size_t i = 0; i < objects_.size(); ++i)
		{
			Object3D* temp = objects_[i];
			//draw hack image if in pick mode
			if (pickMode) temp->drawId(glAccess, eyePoseInverse);
			else temp->draw(glAccess, eyePoseInverse);

		}
		if (!(pickMode || isAnim)) arcball_->draw(glAccess, eyePoseInverse);
	}

	void rotation(const int x, const int y, const double dx, const double dy) {
		if (!doTransformChecks(dx,dy)) return;
		Rbt transform;
		if (currentMut_->getId() == currentEye_->getId() && !(currentMut_->usesArcball() && currentMut_->usingWorldSky())) {
			transform = Rbt(Quaternion::makeXRotation(dy) * Quaternion::makeYRotation(-dx));
		}
		else {
			transform = arcball_->updateRotation(x, y);
		}
		applyTransform(transform);
	}

	void translation(const double dx, const double dy, bool inZ) {
		if (!doTransformChecks(dx,dy)) return;
		Rbt transform = (inZ) ? Rbt(Vector3(0, 0, -dy) * 0.01) : Rbt(Vector3(dx, dy, 0) * 0.01);
		applyTransform(transform);
	}

	bool usingArcball() { return currentMut_->usesArcball();}
	bool setScreenSpaceCircle(const Matrix4& projection,									
								 const double frust_near, const double frust_fovy,			
                                 const int screen_width, const int screen_height) {
		return arcball_->setScreenSpaceCircle(currentEye_->getFrame(), projection, frust_near, frust_fovy, screen_width, screen_height);
	}
	void startRotation(int mouseX, int mouseY) {
		arcball_->startRotation(mouseX, mouseY);
	}
	void switchMode() {
		currentMut_->switchMode();
	}
	void lerpFrames(KeyFrame & f1, KeyFrame & f2, double t) {
		KeyFrame::lerpFrames(f1, f2, t);
		//changeMutable(currentMut_->getId(),0);
		currentEye_ = objects_[eyePoseIndex_];
		//arcball_->setObjectToRotate(currentMut_); we don't care about arcball when lerping
	}
	void startAnim() {
		isAnim = true;
	}
	void stopAnim() {
		isAnim = false;
		changeMutable(currentMut_->getId(),0);
		currentEye_ = objects_[eyePoseIndex_];
		arcball_->setObjectToRotate(currentMut_);
	}

};
