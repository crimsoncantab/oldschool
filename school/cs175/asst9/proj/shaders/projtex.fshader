
uniform sampler2D vTexUnit0;

varying vec4 pTexCoord;
varying vec3 pNormal;
varying vec4 pPosition;


void main(void)
{
	vec3 light = vec3(0,0,0);
	vec3 toLight = normalize(light - pPosition.xyz);

	vec3 normal = normalize(pNormal);

	float diffuse = max(0.0, dot(normal, toLight));

	gl_FragColor = diffuse * texture2D(vTexUnit0, (pTexCoord/pTexCoord.w).xy);
}
