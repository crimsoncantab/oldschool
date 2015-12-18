//
// shader_shells.vert
//
// CS 175: Computer Graphics
// This is the vertex shader for the HelloWorld application.
//

uniform mat4 uProjMatrix;
uniform mat4 uModelViewMatrix;
uniform mat4 uNormalMatrix;
uniform float valpha_exponent;
attribute vec2 vtexCoord;
//attribute vec3 vTangent;

varying vec3 pNormal;
varying vec4 pPosition;
varying vec2 ptexCoord;
varying float palpha_exponent;
//varying vec3 pTangent;

void main()
{
//  pTangent = vec3(uNormalMatrix * normalize(vec4(vTangent.x, vTangent.y, vTangent.z, 0.0)));
  palpha_exponent = valpha_exponent;
  ptexCoord = vtexCoord;
  vec4 normal = normalize(vec4(gl_Normal.x, gl_Normal.y, gl_Normal.z, 0.0));
  pNormal = vec3(uNormalMatrix * normal);
 
  pPosition = uModelViewMatrix * gl_Vertex;
  gl_Position = uProjMatrix * pPosition;
}
