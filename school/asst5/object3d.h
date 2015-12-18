#pragma once
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <GL/glut.h>
#include "matrix4.h"
#include "shaderstate.h"
#include "rbt.h"
#include <vector>
#include <cstddef>
#include <math.h>
using namespace std;
class Object3D //general abstract class for anything in 3D world
{
private:
	static int counter_;
	

protected:
	int id_;
	vector <Object3D*> children_;
	Rbt frame_;
	Rbt beforeChildren_;
	Matrix4 scale_;
	Object3D(string name, Rbt frame): frame_(frame), name_(name){ id_ = getNextId();}
	Object3D(Rbt frame): frame_(frame), name_("a 3d object"){id_ = getNextId(); }
	Object3D(const Object3D& o) : id_(o.id_), frame_(o.frame_), beforeChildren_(o.beforeChildren_), scale_(o.scale_), name_(o.name_) {
		for (size_t i = 0; i < o.children_.size(); ++i)
		{
			children_.push_back(o.children_[i]->clone());
		}
	}

	static int getNextId() { return counter_++;}
	
	static void sendModelViewNormalMatrix(const Matrix4& MVM, ShaderState glAccess)				// takes MVM and sends it (and normal matrix) to the vertex shader
	{
		GLfloat glmatrix[16];
		MVM.writeToColumnMajorMatrix(glmatrix);													// send MVM
		safe_glUniformMatrix4fv(glAccess.h_uModelViewMatrix_, glmatrix);
		
		MVM.getNormalMatrix().writeToColumnMajorMatrix(glmatrix);								// send normal matrix
		safe_glUniformMatrix4fv(glAccess.h_uNormalMatrix_, glmatrix);
	}

	virtual void draw(ShaderState& glAccess){};
	virtual void lerp(Object3D * o1, Object3D * o2, double t) {
		frame_ = o1->frame_.lerp(o2->frame_, t);
		beforeChildren_ = o1->beforeChildren_.lerp(o2->beforeChildren_, t);
		for (size_t i = 0; i < children_.size(); ++i)
		{
			children_[i]->lerp(o1->children_[i], o2->children_[i], t);
		}	
	}

public:
	string name_;
	static float convertIdToColor(int id) {return (float)id/(float)counter_;}
	static int convertColorToId(float color) {return (int)(color * counter_ + .5);}

	int getId() { return id_;}
	void setId(int id) { id_ = id; }
	void preApplyTransformation(Rbt t) {
		frame_ = t * frame_;
	}
	void postApplyTransformation(Rbt t) {
		frame_ = frame_ * t;
	}
	Rbt& getFrame() {
		return frame_;
	}
	//search self and children for object with id.  return the object if it exists, otherwise return null
	Object3D* findId(int id) { 
		if (id == id_) return this; 
		Object3D* temp = NULL;
		for (size_t i = 0; i < children_.size(); ++i)
		{
			temp = children_[i]->findId(id);
			if (temp != NULL) return temp;
		}
		return NULL;
	}
	virtual void draw(ShaderState& glAccess, Rbt& eyePoseInverse) {
		Rbt MVM = eyePoseInverse * frame_;
		sendModelViewNormalMatrix(MVM.getMatrix() * scale_, glAccess);
		draw(glAccess);
		for (size_t i = 0; i < children_.size(); ++i)
		{
			children_[i]->draw(glAccess, MVM * beforeChildren_);
		}		
	}
	//draws id into red, instead of drawing color
	virtual void drawId(ShaderState& glAccess, Rbt& eyePoseInverse){
		Rbt MVM = eyePoseInverse * frame_;
		sendModelViewNormalMatrix(MVM.getMatrix() * scale_, glAccess);
		drawId(glAccess);
		for (size_t i = 0; i < children_.size(); ++i)
		{
			children_[i]->drawId(glAccess, MVM * beforeChildren_);
		}	
	}
	virtual Object3D* clone() = 0;
	virtual void drawId(ShaderState& glAccess) = 0;
	virtual void switchMode() {}
	virtual void addChild(Object3D* child) { children_.push_back(child); }
	virtual void interpretMouseMotion(const int dx, const int dy) {}
	virtual bool usesArcball(){return true;}
	virtual bool usingWorldSky() { return false;}
	virtual bool editableRemotely() {return true; }
	virtual bool usesDefaultTransformBehavior(){return true;}
	virtual bool hasCamera() {return false;}
	virtual bool isEditable() {return true;}
	virtual Object3D * lerp(Object3D * o2, double t) {
		Object3D * lerpation = this->clone();
		lerpation->lerp(this, o2, t);
		return lerpation;
	}
	virtual void save(ostream * out) {
		* out<<id_<<"\n";
		frame_.save(out);
		beforeChildren_.save(out);
		for (size_t i = 0; i < children_.size(); ++i)
		{
			children_[i]->save(out);
		}	
	}
	virtual void load(istream * in) {
		*in >> id_;
		frame_.load(in);
		beforeChildren_.load(in);
		for (size_t i = 0; i < children_.size(); ++i)
		{
			children_[i]->load(in);
		}
	}
};
int Object3D::counter_ = 0;