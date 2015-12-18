//
// hello.frag
//
// Hello World application's fragment program
//  

uniform vec3 uLight;
uniform vec3 uCamera;
uniform samplerCube environMap; //got this from GLSL online

varying float pRefractIndex;
varying vec3 pWorldPos;
varying vec3 pcolorAmbient;
varying vec3 pcolorDiffuse;
varying vec3 pNormal;
varying vec4 pPosition; 

void main(void)
{
vec3 tocamera = normalize(uCamera - pWorldPos);
  vec3 tolight =  normalize(uLight - vec3(pPosition));
  vec3 top = -normalize(vec3(pPosition));
  vec3 h = normalize(top + tolight);
  vec3 normal = normalize(pNormal);
  float specular = pow(max(0.0, dot(h, normal)), 64.0);
  float diffuse = max(0.0, dot(normal, tolight));
  vec3 intensity = vec3(0.1,0.1,0.1) + pcolorDiffuse * diffuse + vec3(0.6,0.6,0.6) * specular;
  float cosRefractAngle = dot(normal, tocamera);
  float cosInAngle =  sqrt(1. - ((pRefractIndex * pRefractIndex)
		* (1. - (cosRefractAngle * cosRefractAngle)))); //using pythagorean identity and snell's law
  vec3 refract = (pRefractIndex * tocamera * -1.) +
		( (pRefractIndex * cosInAngle) - cosRefractAngle) * normal;

  gl_FragColor = (.75 * textureCube(environMap, refract)) + (.25 * vec4(intensity.x, intensity.y, intensity.z, 1.0));
}