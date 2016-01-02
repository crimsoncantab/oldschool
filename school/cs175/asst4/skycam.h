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
	SkyCam(Rbt frame) : Object3D("skycam",frame)
	{
		worldSkyCoordSystem = true;
	}
	void draw(ShaderState& glAccess, Rbt& eyePose) {}
	void switchMode() { worldSkyCoordSystem = !worldSkyCoordSystem;}
	bool usingWorldSky() { return worldSkyCoordSystem; }
	void drawId(ShaderState& glAccess){}
	void drawId(ShaderState& glAccess, Rbt& eyePoseInverse){}
	bool editableRemotely() {return false; }

private:
	bool worldSkyCoordSystem;													//whether to use sky-sky or sky-world coord system
};
