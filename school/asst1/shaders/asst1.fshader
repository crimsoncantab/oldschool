//
// asst1.fshader
//


uniform float VertexScale;
uniform float GradientLoc;	//specifies where the center of the gradient
							//between the two textures is
uniform vec4 ColorMask;	//specifies which colors to filter.

uniform sampler2D texUnit0;
uniform sampler2D texUnit1;

varying float pX;
varying vec2 pTexCoord;

void main(void)
{
	//float lerper = clamp(.3 *VertexScale,0.,1.);
	
	//more of one texture toward the left, more of another toward the right.
	//The "center" of the gradient is determined by GradientLoc
	float lerper = clamp(pTexCoord.x + GradientLoc,0.,1.);

	// create sinusoidal pattern
	//float intensity = 0.2 * sin(pX * 20.0) + 0.5;
	//gl_FragColor = vec4 (intensity, intensity , intensity , 1.0);
	
	// lookup texture color
	vec4 texColor0 = texture2D(texUnit0, pTexCoord);
	vec4 texColor1 = texture2D(texUnit1, pTexCoord);

	vec4 finalTexColor = (lerper)*texColor0 + (1.-lerper)*texColor1;

	// final color is product of sinusoidal and texture colors (modulation)
	//gl_FragColor *= finalTexColor;
	
	//mask the colors
	gl_FragColor = finalTexColor * ColorMask;
}
