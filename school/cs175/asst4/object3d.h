#pragma once
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

public:
	static float convertIdToColor(int id) {return (float)id/(float)counter_;}
	static int convertColorToId(float color) {return (int)(color * counter_ + .5);}
	string name_;
	void preApplyTransformation(Rbt t) {
		frame_ = t * frame_;
	}
	void postApplyTransformation(Rbt t) {
		frame_ = frame_ * t;
	}
	virtual void draw(ShaderState& glAccess, Rbt& eyePoseInverse) {
		Rbt MVM = eyePoseInverse * frame_;
		sendModelViewNormalMatrix(MVM.getMatrix() * scale_, glAccess);
		draw(glAccess);
		for (std::size_t i = 0; i < children_.size(); ++i)
		{
			children_[i]->draw(glAccess, MVM * beforeChildren_);
		}		
	}
	
	//draws id into red, instead of drawing color
	virtual void drawId(ShaderState& glAccess, Rbt& eyePoseInverse){
		Rbt MVM = eyePoseInverse * frame_;
		sendModelViewNormalMatrix(MVM.getMatrix() * scale_, glAccess);
		drawId(glAccess);
		for (std::size_t i = 0; i < children_.size(); ++i)
		{
			children_[i]->drawId(glAccess, MVM * beforeChildren_);
		}	
	}

	virtual void drawId(ShaderState& glAccess) = 0;

	Rbt& getFrame() {
		return frame_;
	}

	virtual bool usesArcball(){return true;}

	virtual void switchMode() {}

	virtual bool usingWorldSky() { return false;}

	virtual bool editableRemotely() {return true; }

	virtual void addChild(Object3D* child) { children_.push_back(child); }

	virtual bool usesDefaultTransformBehavior(){return true;}

	virtual void interpretMouseMotion(const int dx, const int dy) {}

	int getId() { return id_;}
	
	void setId(int id) { id_ = id; }

	//search self and children for object with id.  return the object if it exists, otherwise return null
	Object3D* findId(int id) { 
		if (id == id_) return this; 
		Object3D* temp = NULL;
		for (std::size_t i = 0; i < children_.size(); ++i)
		{
			temp = children_[i]->findId(id);
			if (temp != NULL) return temp;
		}
		return NULL;
	}
};
int Object3D::counter_ = 0;