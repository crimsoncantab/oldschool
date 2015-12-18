#ifndef VEC_H
#define VEC_H

#include <cmath>
#include <cassert>
#include <fstream>


static const double CS175_PI = 3.14159265358979323846264338327950288;


template <typename T, int n> class Vector
{
    T d_[n];

public:
    Vector()														{ for (int i = 0; i < n; ++i) d_[i] = 0; }
    Vector(const T& t)												{ for (int i = 0; i < n; ++i) d_[i] = t; }
    Vector(const T& t0, const T& t1)								{ assert(n==2); d_[0] = t0, d_[1] = t1; }
    Vector(const T& t0, const T& t1, const T& t2)					{ assert(n==3); d_[0] = t0, d_[1] = t1, d_[2] = t2; }
    Vector(const T& t0, const T& t1, const T& t2, const T& t3)		{ assert(n==4); d_[0] = t0, d_[1] = t1, d_[2] = t2, d_[3] = t3; }
    
    static Vector interpolate(const Vector& v0, const Vector& v1, const Vector& v2, const Vector& v3, const double t)
    {
		const double _t(1-t), _t2(_t*_t), _t3(_t2*_t), t2(t*t), t3(t2*t);
//		return v1 * (1-t) + v2 * t;																		// linear interpolation version
		const Vector i1 = v1 + (v2 - v0) * (1/6.0);
		const Vector i2 = v2 - (v3 - v1) * (1/6.0);
		return v1*_t3 + i1*(3*_t2*t) + i2*(3*_t*t2) + v2*t3;											// 3rd order Bezier interpolation
    }
    
    T& operator [] (const int i)									{ return d_[i]; }
    const T& operator [] (const int i) const						{ return d_[i]; }
    
    Vector operator - () const										{ return Vector(*this) *= -1; }
    Vector& operator += (const Vector& v)							{ for (int i = 0; i < n; ++i) d_[i] += v[i]; return *this; }
    Vector& operator -= (const Vector& v)							{ for (int i = 0; i < n; ++i) d_[i] -= v[i]; return *this; }
    Vector& operator *= (const T a)									{ for (int i = 0; i < n; ++i) d_[i] *= a; return *this; }
    Vector& operator /= (const T a)									{ const T inva(1/a); for (int i = 0; i < n; ++i) d_[i] *= inva; return *this; }
    
    Vector operator + (const Vector& v) const						{ return Vector(*this) += v; }
    Vector operator - (const Vector& v) const						{ return Vector(*this) -= v; }
    Vector operator * (const T a) const								{ return Vector(*this) *= a; }
    Vector operator / (const T a) const								{ return std::abs(a) < 1e-10 ? *this : Vector(*this) /= a; }
    
    static Vector cross(const Vector& a, const Vector& b)			{ assert(n==3); return Vector(a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]); }
    static T dot(const Vector& a, const Vector& b)					{ T r(0); for (int i = 0; i < n; ++i) r += a[i]*b[i]; return r; }
    T length2() const												{ return dot(*this, *this); }
    T length() const												{ return std::sqrt(length2()); }
    Vector& normalize()												{ *this /= std::sqrt(length2()); return *this; }
};

template <typename T, int n> static std::ofstream& operator << (std::ofstream& s, const Vector <T, n>& v)
{
	for (int i = 0; i < n; ++i) s << v[i] << (i==n-1 ? "\n" : " ");
	return s;
}
template <typename T, int n> static std::ifstream& operator >> (std::ifstream& s, Vector <T, n>& v)
{
	for (int i = 0; i < n; ++i) s >> v[i];
	return s;
}

typedef Vector <double, 2> Vector2;
typedef Vector <double, 3> Vector3;
typedef Vector <double, 4> Vector4;


#endif
