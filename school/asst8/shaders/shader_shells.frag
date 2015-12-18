//
// shader_shells.frag
//
// Hello World application's fragment program
//  

uniform sampler2D texUnit0;

uniform vec3 uLight;
varying vec3 pNormal;
//varying vec3 pTangent;
varying vec4 pPosition; 
varying vec2 ptexCoord;
varying float palpha_exponent;


void main(void)
{
	vec3 tolight = uLight - vec3(pPosition);
	vec3 top = -normalize(vec3(pPosition));
	tolight = normalize(tolight);
	vec3 h = normalize(top + tolight);
	vec3 normal = normalize(pNormal);

	float u = dot(pNormal/*pTangent*/, tolight);
	float v = dot(pNormal/*pTangent*/, h);
	u = 1.0 - u*u;
	v = pow(1.0 - v*v, 16.0);

	float r = 0.1 + 0.6 * u + 0.3 * v;
	float g = 0.1 + 0.3 * u + 0.3 * v;
	float b = 0.1 + 0.1 * u + 0.3 * v;
	
	float alpha = pow(texture2D(texUnit0, ptexCoord).r, palpha_exponent);

	gl_FragColor = vec4(r, g, b, alpha);
}
