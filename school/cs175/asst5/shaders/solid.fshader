//
// newshader.frag
//



uniform vec3 uLight;

varying vec4 pColor;
varying vec3 pNormal;
varying vec4 pPosition; 


void main()
{
	//vec3 color = pColor + vec4(pNormal*0.0) + pPosition*0.0 + vec4(uLight*0.0);			// we use all variables so compiler doesn't optimize them away
	gl_FragColor = pColor;
}
