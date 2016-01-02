//
// shader_fins.frag
//

uniform sampler2D texUnit1;

uniform vec3 uLight;
varying vec3 pNormal;
varying vec4 pPosition; 
varying vec2 ptexCoord;
varying vec3 pTangent;

void main(void)
{
	vec3 tolight = normalize(uLight - vec3(pPosition));
	vec3 top = -normalize(vec3(pPosition));
	tolight = normalize(tolight);
	vec3 h = normalize(top + tolight);
	vec3 normal = normalize(pNormal);

	float u = dot(pTangent, tolight);
	float v = dot(pTangent, h);

//	float r = 0.05 + 0.6 * (1.0-u*u) + 0.5 * pow(1.0-v*v, 16.0);
//	float g = 0.05 + 0.3 * (1.0-u*u) + 0.5 * pow(1.0-v*v, 16.0);
//	float b = 0.05 + 0.1 * (1.0-u*u) + 0.5 * pow(1.0-v*v, 16.0);
	u = 1.0 - u*u;
	v = pow(1.0 - v*v, 16.0);

	float r = 0.1 + 0.6 * u + 0.3 * v;
	float g = 0.1 + 0.3 * u + 0.3 * v;
	float b = 0.1 + 0.1 * u + 0.3 * v;

	float alpha_fin = max(0.0, 2.0*abs(normal.z) - 1.0);										// this is how alpha is computed for fins
	float alpha_tex = pow(1.0 - texture2D(texUnit1, ptexCoord).r, 1.0 + 10.0*ptexCoord.y);

	gl_FragColor = vec4(r, g, b, alpha_tex*alpha_fin);
}
