
uniform mat4 uProjMatrix;		// camera projection matrix
uniform mat4 uModelViewMatrix;		// camera model view matrix

attribute vec2 vTexCoord;		// texture coordinate

varying vec2  pTexCoord;		// texture coordinates
varying float wInvVal;


void main()
{
	pTexCoord = vTexCoord;
  
	// get vertex position in normalized device coordinates
	gl_Position = uProjMatrix * uModelViewMatrix * gl_Vertex;
	wInvVal = 1.0 / gl_Position.w;
	pTexCoord *= wInvVal;
	gl_Position /= gl_Position.w;
}
