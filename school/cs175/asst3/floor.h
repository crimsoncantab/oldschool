#pragma once

#include "shape.h"
#include <string>
class Floor :
	public Shape //sqare shape, parallel with xy plane
{
public:

	Floor(Vector4 color, const float y, const float size) : Shape("the floor", Rbt(), color), y_(y), size_(size){}

private:
	virtual void draw(ShaderState& glAccess){

		safe_glVertexAttrib4f(glAccess.h_vColor_, color_[0], color_[1], color_[2], color_[3]);					// set color
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
	const float y_;
	const float size_;

};
