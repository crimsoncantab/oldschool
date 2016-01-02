#ifndef	RBT_H
#define RBT_H

#ifdef __MAC__
#	include <GLUT/glut.h>
#else
#	include <GL/glut.h>
#endif
#include <fstream>
#include "matrix4.h"
#include "quaternion.h"


class Rbt
{
	Vector3 t_;																					// t_ component
	Quaternion r_;																				// quaternion r_ component

public:
	Rbt() : t_(0)																		{ assert((Quaternion(1,0,0,0) - r_).norm2() < 1e-8); }
	Rbt(const Vector3& t, const Quaternion& r) : t_(t), r_(r)							{}
	Rbt(const Vector3& t) : t_(t), r_()													{}		// only set translation part (rotation is identity)
	Rbt(const Quaternion& r) : t_(0), r_(r)												{}		// only set rotation part (translation is 0)
	
	static Rbt interpolate(const Rbt& r0, const Rbt& r1, const Rbt& r2, const Rbt& r3, const double t)
	{
		return Rbt(Vector3::interpolate(r0.t_, r1.t_, r2.t_, r3.t_, t), Quaternion::interpolate(r0.r_, r1.r_, r2.r_, r3.r_, t));
	}
	
	Vector3 getTranslation() const														{ return t_; }
	Quaternion getRotation() const														{ return r_; }
	Matrix4	getMatrix() const															{ return Matrix4::makeTranslation(t_) * r_.getMatrix(); }
	Rbt getInverse() const																{ return Rbt(r_.getInverse() * -t_, r_.getInverse()); }

	Rbt& setTranslation(const Vector3& t)												{ t_ = t; return *this; }
	Rbt& setRotation(const Quaternion& r)												{ r_ = r; return *this; }
	
	Vector3 operator * (const Vector3& a) const											{ return t_ + r_*a; }
	Rbt operator * (const Rbt& a) const													{ return Rbt(t_ + r_*a.t_, r_*a.r_); }

	template <class T> void writeToColumnMajorMatrix(const T * const m) const			{ getMatrix().writeToColumnMajorMatrix(m); }
};


static inline std::ofstream& operator << (std::ofstream& s, const Rbt& r)				{ s << r.getTranslation() << r.getRotation(); return s; }
static inline std::ifstream& operator >> (std::ifstream& s, Rbt& r)						{ Vector3 t; Quaternion q; s >> t >> q; r = Rbt(t, q); return s; }


#endif
