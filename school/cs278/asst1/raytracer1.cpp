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

const double hemi_sr = (2.0 * PI);

double light_watt = 1.0;
Vector3 light_c = Vector3(0.0, 10.0, -10.0);
Vector3 light_n = Vector3(1.0, -2.0, 1.0).normalize();
const double light_r = 2.0;

Vector3 plane_n = Vector3(0.0, 1.0, 0.0);
Vector3 plane_p = Vector3(0.0, -10.0, 0.0);
const double plane_f = 1.0 / (2.0 * PI);

Vector3 cam_p = Vector3(0.0, 0.0, 2.0);
Vector3 cam_n = Vector3(0.0, 0.0, -1.0);
const double cam_h = 1.0, cam_w = 1.0;

Vector3 aperture_p = Vector3(0.0);
Vector3 aperture_n = Vector3(0.0, 0.0, -1.0);
const double aperture_r = 0.001;
const double aperture_a = (aperture_r * aperture_r) * PI;
const double cam_ap_dist_sq = (aperture_p - cam_p).len_sq();

const double px_w = cam_w / CAM_RES_X;
const double px_h = cam_h / CAM_RES_Y;
const double px_x_start = cam_p[0] + cam_w / 2 - px_w / 2;
const double px_y_start = cam_p[1] - cam_h / 2 + px_h / 2;
const double px_a = px_w * px_h;

double cos_angle(Vector3 v1, Vector3 v2) {
  return Vector3::dot(v1, v2) / (v1.len() * v2.len());
}

double intersects_plane(Vector3 & point, Vector3 & ray, Vector3 & p_n, Vector3 & p_p) {
  double d = Vector3::dot(p_p * -1, p_n);
  double denom = Vector3::dot(p_n, ray);
  if (denom == 0) return -1;
  return (-d - Vector3::dot(p_n, point)) / denom;
}

bool intersects_cam(Vector3 & point, Vector3 & ray) {
  //light only comes from one side
  if (Vector3::dot(ray, light_n) > 0)
    return false;
  double lambda = intersects_plane(point, ray, light_n, light_c);
  if (lambda < 0)
    return false;
  else
    return Vector3::dist(point + (ray * lambda), light_c) <= light_r;
}

double raytrace(Vector3 pixel) {
  Vector3 ray = aperture_p - pixel;
  double lambda = intersects_plane(pixel, ray, plane_n, plane_p);
  if (lambda < 0) {
    return 0.0;
  } else {
    double const_coeff = std::pow(cos_angle(cam_n, ray), 4) * aperture_a * px_a
        / cam_ap_dist_sq / n; // * (plane_f * hemi_sr) == 1
    //do monte carlo
    double total = 0.0;
    Vector3 point = pixel + (ray * lambda);
    for (int sample = 0; sample < n; ++sample) {
      Vector3 samp_ray = sample_hemisphere(plane_n);
      if (intersects_cam(point, samp_ray))
        total += cos_angle(plane_n, samp_ray);
    }
    return total * const_coeff;
  }
}
volatile double pixels[CAM_RES_Y][CAM_RES_X];

void * run_segment(void * arg) {
  int id = *(int *) arg;
  for (int i = 0; i < CAM_RES_Y; ++i) {
    double y = px_y_start + i * px_w;
    int my_res_x = CAM_RES_X / NTHREADS;
    for (int j = id * my_res_x; j < ((id + 1) * my_res_x); ++j) {
      double x = px_x_start - j * px_w;
      pixels[i][j] = raytrace(Vector3(x, y, cam_p[2]));
    }
  }
}

int main(int argc, char * argv[]) {
  if (argc > 2) {
    std::cerr << "Usage: ./raytrace <num samples>" << std::endl;
  }

  pthread_t threads[NTHREADS];
  int ids[NTHREADS];

  if (argc == 2) n = std::atof(argv[1]);

  for (int t = 0; t < NTHREADS; ++t) {
    ids[t] = t;
    pthread_create(&threads[t], NULL, run_segment, &ids[t]);
  }
  for (int t = 0; t < NTHREADS; ++t) {
    ids[t] = t;
    pthread_join(threads[t], NULL);
  }

  std::cout << CAM_RES_X << std::endl << CAM_RES_Y << std::endl;

  for (int i = 0; i < CAM_RES_Y; ++i) {
    for (int j = 0; j < CAM_RES_X; ++j) {
      std::cout << pixels[i][j] << std::endl;
    }
  }

  return 0;
}
