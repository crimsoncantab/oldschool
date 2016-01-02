#pragma once
#include <string>
#include <GL/glut.h>
#include "matrix4.h"
#include "shaderstate.h"

using namespace std;
class Object3D //general abstract class for anything in 3D world
{
protected:
	Matrix4 frame_;
	Matrix4 auxilaryFrame;
	Object3D(string name, Matrix4 frame): frame_(frame), name_(name){}
	Object3D(Matrix4 frame): frame_(frame), name_("a 3d object"){}
	
	static void sendModelViewNormalMatrix(const Matrix4& MVM, ShaderState glAccess)				// takes MVM and sends it (and normal matrix) to the vertex shader
	{
		GLfloat glmatrix[16];
		MVM.writeToColumnMajorMatrix(glmatrix);													// send MVM
		safe_glUniformMatrix4fv(glAccess.h_uModelViewMatrix_, glmatrix);
		
		MVM.getNormalMatrix().writeToColumnMajorMatrix(glmatrix);								// send normal matrix
		safe_glUniformMatrix4fv(glAccess.h_uNormalMatrix_, glmatrix);
	}

	virtual void draw(ShaderState& glAccess)=0;

public:
	const string name_;
	void preApplyTransformation(Matrix4 t) {
		frame_ = t * frame_;
	}
	void postApplyTransformation(Matrix4 t) {
		frame_ = frame_ * t;
	}
	virtual void draw(ShaderState& glAccess, Matrix4& eyePoseInverse) {
		Matrix4 MVM = eyePoseInverse * frame_;
		sendModelViewNormalMatrix(MVM, glAccess);
		draw(glAccess);
	}

	Matrix4& getFrame() {
		return frame_;
	}

	void setFrame(Matrix4 frame) {
		frame_ = frame;
	}
};
