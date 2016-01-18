#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <pthread.h>

#define NTHREADS 8
#define CAM_RES_X 400
#define CAM_RES_Y 400

static const double PI = 3.14159265358979323846264338327950288;

template <typename T, int n> class Vector {
  T d_[n];

public:

  Vector() {
    for (int i = 0; i < n; ++i) d_[i] = 0;
  }

  Vector(const T& t) {
    for (int i = 0; i < n; ++i) d_[i] = t;
  }

  Vector(const T& t0, const T& t1) {
    assert(n == 2);
    d_[0] = t0, d_[1] = t1;
  }

  Vector(const T& t0, const T& t1, const T& t2) {
    assert(n == 3);
    d_[0] = t0, d_[1] = t1, d_[2] = t2;
  }

  Vector(const T& t0, const T& t1, const T& t2, const T& t3) {
    assert(n == 4);
    d_[0] = t0, d_[1] = t1, d_[2] = t2, d_[3] = t3;
  }

  T& operator [] (const int i) {
    return d_[i];
  }

  const T& operator [] (const int i) const {
    return d_[i];
  }

  Vector operator -() const {
    return Vector(*this) *= -1;
  }

  Vector& operator +=(const Vector& v) {
    for (int i = 0; i < n; ++i) d_[i] += v[i];
    return *this;
  }

  Vector& operator -=(const Vector& v) {
    for (int i = 0; i < n; ++i) d_[i] -= v[i];
    return *this;
  }

  Vector& operator *=(const T a) {
    for (int i = 0; i < n; ++i) d_[i] *= a;
    return *this;
  }

  Vector& operator /=(const T a) {
    const T inva(1 / a);
    for (int i = 0; i < n; ++i) d_[i] *= inva;
    return *this;
  }

  Vector operator +(const Vector& v) const {
    return Vector(*this) += v;
  }

  Vector operator -(const Vector& v) const {
    return Vector(*this) -= v;
  }

  Vector operator *(const T a) const {
    return Vector(*this) *= a;
  }

  Vector operator /(const T a) const {
    return Vector(*this) /= a;
  }

  Vector operator *(const Vector& a) const {
    Vector r;
    for (int i = 0; i < n; ++i) r[i] = d_[i] * a[i];
    return r;
  }

  double len_sq() const {
    return dot(*this, *this);
  }

  double len() const {
    return std::sqrt(len_sq());
  }

  static double dist(const Vector& a, const Vector& b) {
    assert(n == 3);
    return (a - b).len();
  }

  static Vector cross(const Vector& a, const Vector& b) {
    assert(n == 3);
    return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
  }

  static T dot(const Vector& a, const Vector& b) {
    T r(0);
    for (int i = 0; i < n; ++i) r += a[i] * b[i];
    return r;
  }

  Vector& normalize() {
    assert(dot(*this, *this) > 1e-8);
    return *this /= std::sqrt(dot(*this, *this));
  }
};

typedef Vector <double, 3 > Vector3;

Vector3 sample_disk(Vector3& point, Vector3& b1, Vector3& b2, double radius) {
  double x, y;
  do {
    x = 2 * drand48() - 1;
    y = 2 * drand48() - 1;
  } while (x * x + y * y > 1);
  return point + (b1 * x * radius) + (b2 * y * radius);
}

Vector3 sample_hemisphere(Vector3& normal) {
  double u1 = drand48();
  double u2 = drand48();
  double r = std::sqrt(1.0f - u1 * u1);
  double phi = 2 * PI * u2;

  Vector3 v = Vector3(std::cos(phi) * r, std::sin(phi) * r, u1);
  v *= (Vector3::dot(normal, v) < 0 ? -1 : 1);
  return v;
}

volatile int n = 1000;

double hemi_sr = (2.0 * PI);

double light_watt = 1.0;
Vector3 light_c = Vector3(0.0, 10.0, -10.0);
Vector3 light_n = Vector3(1.0, -2.0, 1.0).normalize();
Vector3 light_b1 = Vector3(1.0, 1.0, 1.0).normalize();
Vector3 light_b2 = Vector3(1.0, 0.0, -1.0).normalize();
double light_r = 2.0;
double light_a = (light_r * light_r * PI);

Vector3 plane_n = Vector3(0.0, 1.0, 0.0);
Vector3 plane_p = Vector3(0.0, -10.0, 0.0);
double plane_f = 1.0 / (2.0 * PI);

Vector3 cam_p = Vector3(0.0, 0.0, 2.0);
Vector3 cam_n = Vector3(0.0, 0.0, -1.0);
double cam_h = 1.0, cam_w = 1.0;

Vector3 aperture_p = Vector3(0.0);
Vector3 aperture_n = Vector3(0.0, 0.0, -1.0);
double aperture_r = 0.001;
double aperture_a = (aperture_r * aperture_r) * PI;

double px_w = cam_w / CAM_RES_X;
double px_h = cam_h / CAM_RES_Y;

double cos_angle(Vector3 v1, Vector3 v2) {
  return Vector3::dot(v1, v2) / (v1.len() * v2.len());
}

double intersects_plane(Vector3 & point, Vector3 & ray, Vector3 & p_n, Vector3 & p_p) {
  double d = Vector3::dot(p_p * -1, p_n);
  double denom = Vector3::dot(p_n, ray);
  if (denom == 0) return -1;
  return (-d - Vector3::dot(p_n, point)) / denom;
}

bool intersects_cam(Vector3 & point, /* return param */ int& pixel_x, int& pixel_y) {
  Vector3 ray = aperture_p - point;
  //light only comes in one way
  if (Vector3::dot(ray, aperture_n) > 0)
    return false;
  double lambda = intersects_plane(point, ray, cam_n, cam_p);
  if (lambda < 0)
    return false;
  Vector3 intersect = point + (ray * lambda);

  double diff_x = intersect[0] - cam_p[0];
  double diff_y = intersect[1] - cam_p[1];
  if (std::abs(diff_x) > cam_h / 2 || std::abs(diff_y) > cam_w / 2) {
    return false;
  }
  pixel_x = (cam_w / 2 - diff_x) * CAM_RES_X / cam_w;
  pixel_y = (cam_h / 2 + diff_y) * CAM_RES_Y / cam_h;
  return true;
}

int main(int argc, char * argv[]) {
  if (argc > 2) {
    std::cerr << "Usage: ./raytrace <num samples>" << std::endl;
  }

  if (argc == 2) n = std::atof(argv[1]);
  double pixels[CAM_RES_Y][CAM_RES_X];

  for (int i = 0; i < CAM_RES_Y; ++i) {
    for (int j = 0; j < CAM_RES_X; ++j) {
      pixels[i][j] = 0;
    }
  }

  double const_coeff = plane_f * light_watt * aperture_a / n * hemi_sr * light_a;

  for (int s = 0; s < n; ++s) {
    Vector3 l_pt = sample_disk(light_c, light_b1, light_b2, light_r);
    Vector3 l_ray = sample_hemisphere(light_n);
    double lambda = intersects_plane(l_pt, l_ray, plane_n, plane_p);
    if (lambda < 0)
      continue;
    Vector3 p_pt = l_pt + (l_ray * lambda);
    int px_x, px_y;
    if (intersects_cam(p_pt, px_x, px_y)) {
      Vector3 b_ray = aperture_p - p_pt;
      double watt = cos_angle(-b_ray, aperture_n) / (b_ray).len_sq() *
          cos_angle(l_ray, light_n) * cos_angle(b_ray, plane_n);
      pixels[px_y][px_x] += watt;
    }
  }

  std::cout << CAM_RES_X << std::endl << CAM_RES_Y << std::endl;
  for (int i = 0; i < CAM_RES_Y; ++i) {
    for (int j = 0; j < CAM_RES_X; ++j) {
      std::cout << (pixels[i][j] * const_coeff) << std::endl;
    }
  }

  return 0;
}
