#pragma once

#include <GL/glut.h>
class ShaderState
{
public:
    GLuint h_program_;
    GLint h_uLightE_;
    GLint h_uProjMatrix_;
    GLint h_uModelViewMatrix_;
    GLint h_uNormalMatrix_;
    GLint h_vColor_;
};
