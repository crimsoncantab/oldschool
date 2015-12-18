#ifndef	RBT_H
#define RBT_H

#ifdef __MAC__
#	include <GLUT/glut.h>
#else
#	include <GL/glut.h>
#endif
#include "matrix4.h"
#include "quaternion.h"


class Rbt
{
	Vector3 t_;																					// t_ component
	Quaternion r_;																				// quaternion r_ component

public:
	Rbt() : t_(0)																		{ assert((Quaternion(1, 0,0,0) - r_).norm2() < 1e-8); }
	Rbt(const Vector3& t, const Quaternion& r) : t_(t), r_(r)							{}
	Rbt(const Vector3& t) : t_(t), r_()													{}		// only set translation part (rotation is identity)
	Rbt(const Quaternion& r) : t_(0), r_(r)												{}		// only set rotation part (translation is 0)
	
	Vector3 getTranslation() const														{ return t_; }
	Quaternion getRotation() const														{ return r_; }
	Matrix4	getMatrix() const															{ return Matrix4::makeTranslation(t_) * r_.getMatrix(); }
	Rbt getInverse() const																{ Quaternion rInv = r_.getInverse(); return Rbt(rInv*-t_, rInv); }

	Rbt& setTranslation(const Vector3& t)												{ t_ = t; return *this; }
	Rbt& setRotation(const Quaternion& r)												{ r_ = r; return *this; }
	
	Vector3 operator * (const Vector3& a) const											{ /*did*/return getMatrix() * a; }
	Rbt operator * (const Rbt& a) const													{ /*did*/return Rbt(t_ + r_ * a.t_, r_ * a.r_); }

	template <class T> void writeToColumnMajorMatrix(const T * const m) const			{ getMatrix().writeToColumnMajorMatrix(m); }
};


#endif
