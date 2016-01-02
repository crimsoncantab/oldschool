//
// hello.frag
//
// Hello World application's fragment program
//  

uniform vec3 uLight;

varying vec3 pcolorAmbient;
varying vec3 pcolorDiffuse;
varying vec3 pNormal;
varying vec4 pPosition; 

void main(void)
{
  vec3 tolight = normalize(uLight - vec3(pPosition));
  vec3 normal = normalize(pNormal);

  float diffuse = max(0.0, dot(normal, tolight));
  vec3  intensity = pcolorAmbient + pcolorDiffuse * diffuse;

  gl_FragColor = vec4(intensity.x, intensity.y, intensity.z, 1.0);
}
