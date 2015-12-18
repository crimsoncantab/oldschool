//
// asst1.vshader
//
// CS 175: Computer Graphics
// This is the vertex shader for HelloWorld2D.
// It creates a vertical sinusoidal pattern.
//

uniform  float VertexScale;
uniform  float WindowRatio; // ratio of width and height of window (w/h)

attribute vec2 vTexCoord;

varying float  pX;
varying vec2   pTexCoord;

void main()
{
	//pX = gl_Vertex.x;

	gl_Position = gl_Vertex;
	gl_Position.x *= VertexScale;
	if (WindowRatio > 1) {
		//this window is wide, so reduce the x
		gl_Position.x /= WindowRatio;
	} else {
		//this window is tall, so reduce the y
		//(also includes a ratio of 1, for a square window, which would change nothing)
		gl_Position.y *= WindowRatio;
	}

	pTexCoord = vTexCoord;
}
