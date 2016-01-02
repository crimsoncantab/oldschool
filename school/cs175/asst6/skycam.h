#pragma once
#include <string>
#include "object3d.h"
#include "matrix4.h"
#include "shaderstate.h"
#include <cstddef>

using namespace std;
class SkyCam : //skycam which will not be drawn.
	public Object3D
{
public:
	SkyCam(Rbt frame) : Object3D("Skycam",frame) {worldSkyCoordSystem_ = true;}
	SkyCam(const SkyCam & s) : Object3D(s), worldSkyCoordSystem_(s.worldSkyCoordSystem_) {}
	void draw(ShaderState& glAccess, Rbt& eyePose) {}
	void switchMode() { worldSkyCoordSystem_ = !worldSkyCoordSystem_;}
	bool usingWorldSky() { return worldSkyCoordSystem_; }
	void drawId(ShaderState& glAccess){}
	void drawId(ShaderState& glAccess, Rbt& eyePoseInverse){}
	bool editableRemotely() {return false; }
	bool hasCamera() {return true;}

private:
	bool worldSkyCoordSystem_;													//whether to use sky-sky or sky-world coord system
protected:
	virtual Object3D* clone() { return new SkyCam(*this);}
};
