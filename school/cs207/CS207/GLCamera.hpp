#ifndef CS207_GLCAMERA_HPP
#define CS207_GLCAMERA_HPP

/**
 * @file GLCamera.hpp
 * Provides a wrapper for the gluLookAt command
 *
 * @brief Keeps track of orientation, zoom, and view point to compute
 * the correct gluLookAt command. This provides more intuitive commands
 * such as pan, zoom, rotate, etc.
 */

#include <math.h>
#include "Point.hpp"

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#endif

namespace CS207
{

/**
 * View point with vectors defining the local axis
 *
 *              upV
 *               |  rightV
 *               | /
 *               |/
 *               * point
 *                \
 *                 \
 *                  eyeV
 */
class GLCamera {
 private:

  float dist;

  Point point;
  Point upV;
  Point rightV;
  Point eyeV;

 public:

  /**
   * Constructor
   */
  GLCamera()
    : dist(1), point(0, 0, 0), upV(0, 0, 1), rightV(0, 1, 0), eyeV(1, 0, 0) {
  }

  /**
   * Set the OpenGL ModelView matrix to the Camera's orientation and axis
   */
  inline void set_GLView() const {
    Point eye = point + eyeV * dist;

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(eye.x, eye.y, eye.z,
	      point.x, point.y, point.z,
	      upV.x, upV.y, upV.z);
  }

  /**
   * Change the point we are viewing with respect to the local axes
   */
  inline void pan(float x, float y, float z) {
    point = point + rightV * x;
    point = point + upV    * y;
    point = point + eyeV   * z;
  }

  /**
   * Set the new view point. The camera now looks at (x,y,z)
   */
  inline void view_point(float x, float y, float z) {
    point = Point(x, y, z);
  }
  /** @overload */
  inline void view_point(const Point& p) {
    point = p;
  }

  /**
   * Rotate the view about the local x-axis
   */
  inline void rotate_x(float angle) {
    // Rotate eye vector about the right vector
    eyeV = eyeV * cosf(angle) + upV * sinf(angle);
    // Normalize for numerical stability
    eyeV.normalize();

    // Compute the new up vector by cross product
    upV = eyeV.cross(rightV);
    // Normalize for numerical stability
    upV.normalize();
  }

  /**
   * Rotate the view about the local y-axis
   */
  inline void rotate_y(float angle) {
    // Rotate eye about the up vector
    eyeV = eyeV * cosf(angle) + rightV * sinf(angle);
    // Normalize for numerical stability
    eyeV.normalize();

    // Compute the new right vector by cross product
    rightV = upV.cross(eyeV);
    // Normalize for numerical stability
    rightV.normalize();
  }

  /**
   * Zoom by a scale factor
   */
  inline void zoom(float d) {
    dist *= d;
  }

  /**
   * Set the zoom magnitude
   */
  inline void zoom_mag(float d) {
    dist = d;
  }
}; // end GLCamera

} // end namespace CS207
#endif
