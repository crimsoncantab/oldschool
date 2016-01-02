//
// hello.frag
//
// Hello World application's fragment program
//  

uniform vec3 uLight;
uniform vec3 uCamera;
uniform samplerCube environMap; //got this from GLSL online

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
  vec3 reflect = 2 * dot(normal, tocamera) * normal - tocamera;
  gl_FragColor = (.75 * textureCube(environMap, reflect)) + (.25 * vec4(intensity.x, intensity.y, intensity.z, 1.0));
}
