//
// This does correct projective texturing using the 
// graphics hardware
//


uniform mat4 uProjMatrix;		// camera projection matrix
uniform mat4 uModelViewMatrix;		// camera model view matrix

attribute vec2 vTexCoord;		// texture coordinate

varying vec2  pTexCoord;		// texture coordinates

void main()
{
	pTexCoord = vTexCoord;
  
	// get vertex position in normalized device coordinates
	gl_Position = uProjMatrix * uModelViewMatrix * gl_Vertex;
}
