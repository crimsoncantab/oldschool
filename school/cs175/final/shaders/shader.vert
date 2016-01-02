//
// hello.vert
//
// CS 175: Computer Graphics
// This is the vertex shader for the HelloWorld application.
//

uniform mat4 uProjMatrix;
uniform mat4 uModelViewMatrix;
uniform mat4 uNormalMatrix;

attribute vec3 colorAmbient;
attribute vec3 colorDiffuse;

varying vec3 pcolorAmbient;
varying vec3 pcolorDiffuse;
varying vec3 pNormal;
varying vec4 pPosition;

void main()
{
  pcolorAmbient = colorAmbient;
  pcolorDiffuse = colorDiffuse;
  vec4 normal = normalize(vec4(gl_Normal.x, gl_Normal.y, gl_Normal.z, 0.0));
  pNormal = vec3(uNormalMatrix * normal);
   
  pPosition = uModelViewMatrix * gl_Vertex;
  gl_Position = uProjMatrix * pPosition;
}
