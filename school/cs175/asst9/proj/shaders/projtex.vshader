
uniform mat4 uProjMatrix;		// camera projection matrix
uniform mat4 uModelViewMatrix;		// camera model view matrix

uniform mat4 uProjMatrixPjtr;		// projector projection matrix
uniform mat4 uModelViewMatrixPjtr;	// projector model view matrix

uniform mat4 uNormalMatrixPjtr;		// for getting normals wrt projector's view

varying vec3  pNormal;			// normal in projector's coord sys
varying vec4  pPosition; 		// position in projector's coord sys
varying vec4  pTexCoord;		// texture coordinates

void main()
{
	// first compute normal in projector's coord sys
	vec4 normal = vec4(gl_Normal.x, gl_Normal.y, gl_Normal.z, 1.0);
	pNormal = vec3(uNormalMatrixPjtr * normal);

	// position of point in projector's coord sys
	pPosition=  uModelViewMatrixPjtr * gl_Vertex;

	// get texture coordinates
	pTexCoord= uProjMatrixPjtr * uModelViewMatrixPjtr * gl_Vertex;
	
	// get vertex position in normalized device coordinates
	gl_Position = uProjMatrix * uModelViewMatrix * gl_Vertex;
}
