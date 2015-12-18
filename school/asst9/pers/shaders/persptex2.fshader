
uniform sampler2D vTexUnit0;

varying vec2  pTexCoord;		// texture coordinates
varying float wInvVal;

void main()
{
	gl_FragColor = texture2D(vTexUnit0, pTexCoord / wInvVal);
}
