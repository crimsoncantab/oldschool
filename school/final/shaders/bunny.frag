//
// hello.frag
//
// Hello World application's fragment program
//  

uniform vec3 uLight;
uniform vec3 uCamera;
uniform samplerCube environMap; //got this from GLSL online

varying vec3 pcolorAmbient;
varying vec3 pcolorDiffuse;
varying vec3 pNormal;
varying vec4 pPosition; 

void main(void)
{
  vec3 tolight = normalize(uLight - vec3(pPosition));
  vec3 normal = normalize(pNormal);
  gl_FragColor = (textureCube(environMap, normal));// + (.75 * vec4(intensity.x, intensity.y, intensity.z, 1.0));
}
