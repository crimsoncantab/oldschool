#ifndef CS207_GLTOOLS_HH
#define	CS207_GLTOOLS_HH

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

// Map C++ types to GL types at compilation time: gltype<T>::value

template <typename T> struct gltype {
};

template <GLenum V> struct gltype_value {
  static constexpr GLenum value = V;
};

template <> struct gltype<unsigned char>
: public gltype_value<GL_UNSIGNED_BYTE> {
};

template <> struct gltype<unsigned short>
: public gltype_value<GL_UNSIGNED_SHORT> {
};

template <> struct gltype<unsigned int>
: public gltype_value<GL_UNSIGNED_INT> {
};

template <> struct gltype<short>
: public gltype_value<GL_SHORT> {
};

template <> struct gltype<int>
: public gltype_value<GL_INT> {
};

template <> struct gltype<float>
: public gltype_value<GL_FLOAT> {
};

template <> struct gltype<double>
: public gltype_value<GL_DOUBLE> {
};


namespace GLTools {

void init_simple_light() {

  GLfloat mat_specular[] = {1.0, 1.0, 1.0, 0};
  GLfloat mat_shininess[] = {50.0};
  GLfloat light_position[] = {0, 0, 1.5, 0};
  GLfloat light_ambient[] = {.2, .2, .2, 1.0};
  GLfloat light_diffuse[] = {1, 1, 1, 1.0};
  GLfloat light_specular[] = {1, 1, 1, 1.0};

  glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
  glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
  glLightfv(GL_LIGHT0, GL_POSITION, light_position);
  glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
  glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);

  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_LIGHT0);
}

void init_point_dist() {
  float iPointSizeCoords[3] = {0, 1, 1};
  glPointParameterf(GL_POINT_SIZE_MIN, .1f);
  glPointParameterf(GL_POINT_SIZE_MAX, 100.0f);
  glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, iPointSizeCoords);

}

/** Print any outstanding OpenGL error to std::cerr. */
void check_gl_error(const char *context = "") {
  GLenum errCode = glGetError();
  ;
  if (errCode != GL_NO_ERROR) {
    const GLubyte *error = gluErrorString(errCode);
    std::cerr << "OpenGL error";
    if (context && *context)
      std::cerr << " at " << context;
    std::cerr << ": " << error << "\n";
  }
}


}

#endif	/* CS207_GLTOOLS_HH */

