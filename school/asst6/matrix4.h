#ifndef MATRIX4_H
#define MATRIX4_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#ifdef __MAC__
#	include <GLUT/glut.h>
#else
#	include <GL/glut.h>
#endif
#include <cmath>
#include "vec.h"

using namespace std;
class Matrix4
{
	double d_[16];																							// layout is row-major

public:
	double * operator [] (const int i)										{ return &d_[i << 2]; }
	const double * operator [] (const int i) const							{ return &d_[i << 2]; }
	bool is_affine() const													{ return std::abs(d_[15]-1) + std::abs(d_[14]) + std::abs(d_[13]) + std::abs(d_[12]) < 1e-6; }
	double norm2() const													{ double r(0); for (int i = 0; i < 16; ++i) r += d_[i]*d_[i]; return r; }

	Matrix4()																{ std::memset(d_, 0, sizeof(double)*16); for (int i = 0; i < 4; ++i) (*this)[i][i] = 1; }
	Matrix4(const double a)													{ for (int i = 0; i < 16; ++i) d_[i] = a; }
	template <class T> Matrix4& readFromColumnMajorMatrix(const T m[])		{ for (int i = 0; i < 16; ++i) d_[i] = m[i]; return *this = getTranspose(); }
	template <class T> void writeToColumnMajorMatrix(T m[]) const			{ Matrix4 t(getTranspose()); for (int i = 0; i < 16; ++i) m[i] = t.d_[i]; }

	Matrix4& operator += (const Matrix4& a)									{ for (int i = 0; i < 16; ++i) d_[i] += a.d_[i]; return *this; }
	Matrix4& operator -= (const Matrix4& a)									{ for (int i = 0; i < 16; ++i) d_[i] -= a.d_[i]; return *this; }
	Matrix4& operator *= (const double a)									{ for (int i = 0; i < 16; ++i) d_[i] *= a; return *this; }
	Matrix4& operator *= (const Matrix4& a)									{ return *this = *this * a; }
	
	Matrix4 operator + (const Matrix4& a) const								{ return Matrix4(*this) += a; }
	Matrix4 operator - (const Matrix4& a) const								{ return Matrix4(*this) -= a; }
	Matrix4 operator * (const double a) const								{ return Matrix4(*this) *= a; }
	Vector4 operator * (const Vector4& a) const								{ Vector4 r(0); for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) r[i] += (*this)[i][j] * a[j]; return r; }
	Vector3 operator * (const Vector3& a) const
	{
		return Vector3((*this)[0][0]*a[0] + (*this)[0][1]*a[1] + (*this)[0][2]*a[2] + (*this)[0][3],
					   (*this)[1][0]*a[0] + (*this)[1][1]*a[1] + (*this)[1][2]*a[2] + (*this)[1][3],
					   (*this)[2][0]*a[0] + (*this)[2][1]*a[1] + (*this)[2][2]*a[2] + (*this)[2][3]);
	}
	Matrix4 operator * (const Matrix4& a) const
	{
		Matrix4 r(0);
		for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
		for (int k = 0; k < 4; ++k) r[i][k] += (*this)[i][j] * a[j][k];
		return r;
	}

	Matrix4 getAffineInverse() const																		// computes inverse of affine matrix. assumes last row is [0,0,0,1]
	{
		Matrix4 r;																							// default constructor initializes it to identity
		assert(is_affine());
		double det = (*this)[0][0]*((*this)[1][1]*(*this)[2][2] - (*this)[1][2]*(*this)[2][1]) + 
					 (*this)[0][1]*((*this)[1][2]*(*this)[2][0] - (*this)[1][0]*(*this)[2][2]) + 
					 (*this)[0][2]*((*this)[1][0]*(*this)[2][1] - (*this)[1][1]*(*this)[2][0]);
		assert(std::abs(det) > 1e-8);																		// check non-singular matrix
    	r[0][0] =  ((*this)[1][1] * (*this)[2][2] - (*this)[1][2] * (*this)[2][1]) / det;					// "rotation part"
    	r[1][0] = -((*this)[1][0] * (*this)[2][2] - (*this)[1][2] * (*this)[2][0]) / det;
    	r[2][0] =  ((*this)[1][0] * (*this)[2][1] - (*this)[1][1] * (*this)[2][0]) / det;
    	r[0][1] = -((*this)[0][1] * (*this)[2][2] - (*this)[0][2] * (*this)[2][1]) / det;
    	r[1][1] =  ((*this)[0][0] * (*this)[2][2] - (*this)[0][2] * (*this)[2][0]) / det;
    	r[2][1] = -((*this)[0][0] * (*this)[2][1] - (*this)[0][1] * (*this)[2][0]) / det;
    	r[0][2] =  ((*this)[0][1] * (*this)[1][2] - (*this)[0][2] * (*this)[1][1]) / det;
    	r[1][2] = -((*this)[0][0] * (*this)[1][2] - (*this)[0][2] * (*this)[1][0]) / det;
    	r[2][2] =  ((*this)[0][0] * (*this)[1][1] - (*this)[0][1] * (*this)[1][0]) / det;
    
    	r[0][3] = -((*this)[0][3] * r[0][0] + (*this)[1][3] * r[0][1] + (*this)[2][3] * r[0][2]);			// "translation part" - multiply the translation (on the left) by the inverse linear part
    	r[1][3] = -((*this)[0][3] * r[1][0] + (*this)[1][3] * r[1][1] + (*this)[2][3] * r[1][2]);
    	r[2][3] = -((*this)[0][3] * r[2][0] + (*this)[1][3] * r[2][1] + (*this)[2][3] * r[2][2]);		
		assert(r.is_affine() && (Matrix4() - *this*r).norm2() < 1e-8);
		return r;
	}
	Matrix4 getTranspose() const											{ Matrix4 r(0); for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) r[i][j] = (*this)[j][i]; return r; }
	Matrix4	getNormalMatrix() const											{ Matrix4 inv(getAffineInverse()); inv[0][3] = inv[1][3] = inv[2][3] = 0; return inv.getTranspose(); }

	static Matrix4 makeXRotation(const double ang)							{ return makeXRotation(std::cos(ang * CS175_PI/180), std::sin(ang * CS175_PI/180)); }
	static Matrix4 makeYRotation(const double ang)							{ return makeYRotation(std::cos(ang * CS175_PI/180), std::sin(ang * CS175_PI/180)); }
	static Matrix4 makeZRotation(const double ang)							{ return makeZRotation(std::cos(ang * CS175_PI/180), std::sin(ang * CS175_PI/180)); }
	static Matrix4 makeXRotation(const double c, const double s)			{ Matrix4 r; r[1][1] = r[2][2] = c; r[1][2] = -s; r[2][1] = s; return r; }
	static Matrix4 makeYRotation(const double c, const double s)			{ Matrix4 r; r[0][0] = r[2][2] = c; r[0][2] = s; r[2][0] = -s; return r; }
	static Matrix4 makeZRotation(const double c, const double s)			{ Matrix4 r; r[0][0] = r[1][1] = c; r[0][1] = -s; r[1][0] = s; return r; }
	static Matrix4 makeTranslation(const Vector3& t)						{ Matrix4 r; for (int i = 0; i < 3; ++i) r[i][3] = t[i]; return r; }
	static Matrix4 makeScale(const Vector3& s)								{ Matrix4 r; for (int i = 0; i < 3; ++i) r[i][i] = s[i]; return r; }

	static Matrix4 makeProjection(const double top, const double bottom, const double left, const double right, const double near__, const double far__)
	{
		Matrix4 r(0);
		if (std::abs(right - left) > 1e-8)																	// 1st row
		{
			r[0][0] = -2.0 * near__ / (right - left);
			r[0][2] = (right+left) / (right - left);
		}
		if(std::abs(top - bottom) > 1e-8)																	// 2nd row
		{
			r[1][1] = -2.0 * near__ / (top - bottom);
			r[1][2] = (top + bottom) / (top - bottom);
		}
		if(std::abs(far__ - near__) > 1e-8)																	// 3rd row
		{
			r[2][2] = (far__+near__) / (far__ - near__);
			r[2][3] = -2.0 * far__ * near__ / (far__ - near__);
		}
		r[3][2] = -1.0;
		return r;
	}
	static Matrix4 makeProjection(const double fovy, const double aspect_ratio, const double zNear, const double zFar)
	{
		Matrix4 r(0);
		const double ang = fovy * 0.5 * CS175_PI/180;
		const double f = std::abs(std::sin(ang)) < 1e-8 ? 0 : 1/std::tan(ang);
		if (std::abs(aspect_ratio) > 1e-8) r[0][0] = aspect_ratio > 1 ? f/aspect_ratio : f;					// 1st row
		r[1][1] = aspect_ratio > 1 ? f : f*aspect_ratio;													// 2nd row
		if (std::abs(zFar - zNear) > 1e-8)																	// 3rd row
		{
			r[2][2] = (zFar+zNear) / (zFar - zNear);
			r[2][3] = -2.0 * zFar * zNear / (zFar - zNear);
		}
		r[3][2] = -1.0;
		return r;
	}
};



#endif 

