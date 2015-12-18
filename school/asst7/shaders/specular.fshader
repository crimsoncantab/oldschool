

uniform vec3 uLight;


varying vec3 pColor;
varying vec3 pNormal;
varying vec4 pPosition; 

void main()
{
	vec3 tolight = uLight - vec3(pPosition);
	vec3 top = -normalize(vec3(pPosition));
	tolight = normalize(tolight);
	vec3 h = normalize(top + tolight);
	vec3 normal = normalize(pNormal);

	float specular = pow(max(0.0, dot(h, normal)), 64.0);									// higher exponent makes it "shinier"
	float diffuse = max(0.0, dot(normal, tolight));											// diffuse component
	vec3 intensity = vec3(0.1,0.1,0.1) + pColor * diffuse + vec3(0.6,0.6,0.6) * specular;	// combine the two

	gl_FragColor = vec4(intensity.x, intensity.y, intensity.z, 1.0);
}
