#pragma once
#include <string>
#include "object3d.h"
#include "matrix4.h"
#include "shaderstate.h"

using namespace std;
class SkyCam : //skycam which will not be drawn.
	public Object3D
{
public:
	SkyCam(Matrix4 frame) : Object3D("skycam",frame){}

private:
	void draw(ShaderState& glAccess) {}
	void draw(ShaderState& glAccess, Matrix4& eyePose) {}
};
