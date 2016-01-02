#ifndef	QUATERNION_H
#define QUATERNION_H

#ifdef __MAC__
#	include <GLUT/glut.h>
#else
#	include <GL/glut.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "matrix4.h"
#include <cmath>


using namespace std;
class Quaternion
{
	Vector4 q_;																										// layout is: q_[0]==w, q_[1]==x, q_[2]==y, q_[3]==z

public:
	double operator [] (const int i) const														{ return q_[i]; }
	double& operator [] (const int i)															{ return q_[i]; }
	double norm2() const																		{ return Vector4::dot(q_, q_); }

	Quaternion() : q_(1,0,0,0)																	{}
	Quaternion(const double w, const Vector3& v) : q_(w, v[0], v[1], v[2])						{}
	Quaternion(const double w, const double x, const double y, const double z) : q_(w, x,y,z)	{}

	Quaternion& normalize()																		{ q_.normalize(); return *this; }
	Quaternion& operator += (const Quaternion& a)												{ q_ += a.q_; return *this; }
	Quaternion& operator -= (const Quaternion& a)												{ q_ -= a.q_; return *this; }
	Quaternion& operator *= (const double a)													{ q_ *= a; return *this; }

	Quaternion operator + (const Quaternion& a) const											{ return Quaternion(*this) += a; }
	Quaternion operator - (const Quaternion& a) const											{ return Quaternion(*this) -= a; }
	Quaternion operator * (const double a) const												{ return Quaternion(*this) *= a; }
	Quaternion power(const double t) const
	{
		Vector4 q__ = (q_[0] < 0) ? q_ * -1 : q_;
		Vector3 axis(q__[1], q__[2], q__[3]);
		double alphat = atan2(axis.length(), q__[0]) * t;
		return Quaternion(std::cos(alphat), ((axis.length() < 1e-4)?axis:axis.normalize()) * std::sin(alphat)); 

	}
	Quaternion operator * (const Quaternion& a) const
	{
		const Vector3 u(q_[1],q_[2],q_[3]), v(a.q_[1],a.q_[2],a.q_[3]);
		return Quaternion(q_[0]*a.q_[0] - Vector3::dot(u, v), (v*q_[0] + u*a.q_[0]) + Vector3::cross(u, v));
	}
	Vector3 operator * (const Vector3& a) const													{ const Quaternion r = *this * (Quaternion(0, a) * getInverse()); return Vector3(r[1], r[2], r[3]); }

	static double dot(const Quaternion& a, const Quaternion& b)									{ return Vector4::dot(a.q_, b.q_); }
	template <class T> void writeToColumnMajorMatrix(T * m[])									{ getMatrix().writeToColumnMajorMatrix(m); }

	Quaternion getInverse() const																{ const double n = norm2(); assert(n > 1e-8); return Quaternion(q_[0], -q_[1], -q_[2], -q_[3]) * (1/n); }
	Matrix4 getMatrix() const
	{
		Matrix4 r;
		const double n = norm2();
		if (n < 1e-8) return Matrix4(0);
		r[0][0] -= (q_[2]*q_[2] + q_[3]*q_[3]) * (2/n);
		r[0][1] += (q_[1]*q_[2] - q_[0]*q_[3]) * (2/n);
		r[0][2] += (q_[1]*q_[3] + q_[2]*q_[0]) * (2/n);
		r[1][0] += (q_[1]*q_[2] + q_[0]*q_[3]) * (2/n);
		r[1][1] -= (q_[1]*q_[1] + q_[3]*q_[3]) * (2/n);
		r[1][2] += (q_[2]*q_[3] - q_[1]*q_[0]) * (2/n);
		r[2][0] += (q_[1]*q_[3] - q_[2]*q_[0]) * (2/n);
		r[2][1] += (q_[2]*q_[3] + q_[1]*q_[0]) * (2/n);
		r[2][2] -= (q_[1]*q_[1] + q_[2]*q_[2]) * (2/n);
		assert(r.is_affine());
		return r;
	}
	
	Quaternion lerp(Quaternion q2, double t) {
		Quaternion myInverse = getInverse();
		Quaternion ret = ((q2 * myInverse).power(t)) * (*this);
		return ret;
	}
	Quaternion bezerp(Quaternion q2, Quaternion prev, Quaternion post, double t) {
		Quaternion vals[4];
		Quaternion prevInverse = prev.getInverse();
		Quaternion myInverse = getInverse();
		vals[0] = (*this);
		vals[1] = (q2 * prevInverse).power(1./6.) * (*this);
		Quaternion interim = (post * myInverse);
		Quaternion interimPower = interim.power(1./6.);
		if (vals[1].q_[0] < 0) vals[1] *= -1;
		vals[2] = (q2.getInverse() * interimPower).getInverse();
		if (vals[2].q_[0] < 0) vals[2] *= -1;
		vals[3] = q2;

		for (int i = 3; i > 0; i--) {
			for (int j = 0; j < i; j++) {
				vals[j] = vals[j].lerp(vals[j+1], t);
			}
		}
		return vals[0];
	}
	void save(ostream * out) {
		q_.save(out);
	}
	static Quaternion load(istream * in) {
		double w;
		double x;
		double y;
		double z;
		*in >> w;
		*in >> x;
		*in >> y;
		*in >> z;
		return Quaternion(w,x,y,z);
	}
	static Quaternion makeXRotation(const double ang)											{ Quaternion r; const double h = 0.5 * ang * CS175_PI/180; r.q_[1] = std::sin(h); r.q_[0] = std::cos(h); return r; }
	static Quaternion makeYRotation(const double ang)											{ Quaternion r; const double h = 0.5 * ang * CS175_PI/180; r.q_[2] = std::sin(h); r.q_[0] = std::cos(h); return r; }
	static Quaternion makeZRotation(const double ang)											{ Quaternion r; const double h = 0.5 * ang * CS175_PI/180; r.q_[3] = std::sin(h); r.q_[0] = std::cos(h); return r; }
};


#endif
