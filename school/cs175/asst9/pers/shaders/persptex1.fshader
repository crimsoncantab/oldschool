//
// This does correct projective texturing using the 
// graphics hardware
//

uniform sampler2D vTexUnit0;

varying vec2  pTexCoord;		// texture coordinates

void main()
{
	vec4 texColor0 = texture2D(vTexUnit0, pTexCoord);
	gl_FragColor = texColor0;
}
