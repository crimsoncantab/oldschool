// GDC
// Guille Canas. 2008

#ifndef VEC_H
#define VEC_H

#include <cmath>
#include <cassert>


static const double MY_PI = 3.14159265358979323846264338327950288;

template <typename T, int n> class vec_t
{
    T d_[n];

public:
    vec_t()															{}
    vec_t(const T& t)												{ for (int i = 0; i < n; ++i) d_[i] = t; }
    vec_t(const T& t0, const T& t1)									{ assert(n==2); d_[0] = t0, d_[1] = t1; }
    vec_t(const T& t0, const T& t1, const T& t2)					{ assert(n==3); d_[0] = t0, d_[1] = t1, d_[2] = t2; }
    vec_t(const T& t0, const T& t1, const T& t2, const T& t3)		{ assert(n==4); d_[0] = t0, d_[1] = t1, d_[2] = t2, d_[3] = t3; }
    
    T& operator [] (const int i)									{ return d_[i]; }
    const T& operator [] (const int i) const						{ return d_[i]; }
    
    vec_t& operator += (const vec_t& v)								{ for (int i = 0; i < n; ++i) d_[i] += v[i]; return *this; }
    vec_t& operator -= (const vec_t& v)								{ for (int i = 0; i < n; ++i) d_[i] -= v[i]; return *this; }
    vec_t& operator *= (const vec_t& v)								{ for (int i = 0; i < n; ++i) d_[i] *= v[i]; return *this; }
    vec_t& operator /= (const vec_t& v)								{ for (int i = 0; i < n; ++i) d_[i] /= v[i]; return *this; }
    vec_t& operator *= (const T a)									{ for (int i = 0; i < n; ++i) d_[i] *= a; return *this; }
    vec_t& operator /= (const T a)									{ const T inva(1/a); for (int i = 0; i < n; ++i) d_[i] *= inva; return *this; }
    
    vec_t operator + (const vec_t& v) const							{ return vec_t(*this) += v; }
    vec_t operator - (const vec_t& v) const							{ return vec_t(*this) -= v; }
    vec_t operator * (const vec_t& v) const							{ return vec_t(*this) *= v; }
    vec_t operator / (const vec_t& v) const							{ return vec_t(*this) /= v; }
    vec_t operator * (const T a) const								{ return vec_t(*this) *= a; }
    vec_t operator / (const T a) const								{ return vec_t(*this) /= a; }
    
    static vec_t cross(const vec_t& a, const vec_t& b)				{ assert(n==3); return vec_t(a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]); }
    static T dot(const vec_t& a, const vec_t& b)					{ T r(0); for (int i = 0; i < n; ++i) r += a[i]*b[i]; return r; }
    T dist2(const vec_t& a) const									{ const vec_t d(a - *this); return dot(d, d); }
    T dist(const vec_t& a) const									{ return std::sqrt(dist2(a)); }
    T length2() const												{ return dot(*this, *this); }
    T length() const												{ return std::sqrt(length2()); }
    vec_t& normalize()												{ return *this /= std::sqrt(dot(*this, *this)); }

	template <int i, int j> vec_t& rotate(const T& c, const T& s)
	{
		const T t = d_[i];
		d_[i] = t * c - d_[j] * s;
		d_[j] *= c;
		d_[j] += t * s;
		return *this;
	}
	template <int i, int j> vec_t& rotate(const T& a)				{ return rotate <i,j> (std::cos(a), std::sin(a)); }
};

typedef vec_t <double, 2> Vector2;
typedef vec_t <double, 3> Vector3;
typedef vec_t <double, 4> Vector4;


#endif
