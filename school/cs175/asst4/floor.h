#pragma once

#include "shape.h"
#include <string>
#include <cstddef>
class Floor :
	public Shape //sqare shape, parallel with xy plane
{
public:
	Floor(Vector4 color, const float y, const float size) : Shape("the floor", Rbt(), color), y_(y), size_(size){}

private:
	virtual void drawShape(ShaderState& glAccess){
		glBegin(GL_TRIANGLES);
		glNormal3f(0.0, 1.0, 0.0);
		glVertex3f(-size_, y_, -size_);
		glNormal3f(0.0, 1.0, 0.0);
		glVertex3f( size_, y_,  size_);
		glNormal3f(0.0, 1.0, 0.0);
		glVertex3f( size_, y_, -size_);
		
		glNormal3f(0.0, 1.0, 0.0);
		glVertex3f(-size_, y_, -size_);
		glNormal3f(0.0, 1.0, 0.0);
		glVertex3f(-size_, y_,  size_);
		glNormal3f(0.0, 1.0, 0.0);
		glVertex3f( size_, y_,  size_);
		glEnd();
	}
	virtual void drawId(ShaderState& glAccess, Rbt& eyePoseInverse) {}
	const float y_;
	const float size_;

};
