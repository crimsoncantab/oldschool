////////////////////////////////////////////////////////////////////////
//
//   Harvard Computer Science
//	 CS 175: Computer Graphics
//   Professor Steven Gortler
//
//   File: HelloWorld.cpp
//   Name: Loren McGinnis
//   Date: 9/11/2009
//   Desc: Assignment 1
//
////////////////////////////////////////////////////////////////////////

// I N C L U D E S /////////////////////////////////////////////////////

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <GL/glew.h>
#ifdef __MAC__
#	include <GLUT/glut.h>
#else
#	include <GL/glut.h>
#endif
#include "ppm.h"
#include "shader.h"


// G L O B A L S ///////////////////////////////////////////////////////

// application globals
static float		g_red_mask			= 1.;
static float		g_green_mask			= 1.;
static float		g_blue_mask			= 1.;
static int		g_width				= 512;	/// screen width
static int		g_height			= 512;	/// screen height
static bool	g_left_clicked		= false;/// is the left mouse button down?
static bool	g_right_clicked		= false;/// is the right mouse button down?
static float	g_obj_scale			= 1.0;	/// scale factor for object
static float	g_gradient_loc		= 0.0;	/// how much of each texture shows up. 0 is an equal amount.
static int		g_lclick_x, g_lclick_y;		/// coordinates for mouse click event
static int		g_rclick_x, g_rclick_y;		/// coordinates for mouse click event

// OpenGL non-shader handles
static GLuint	h_texture;					/// handle to texture
static GLuint	h_texture2;			        /// handle to second texture

// shader globals
static GLuint	h_program;				/// handle to OpenGL program object
static GLuint  h_vTexCoord;			/// handle to attribute var for texture coordinates
static GLint	h_texUnit0;				/// handle to texture unit variable in fragment program
static GLint	h_texUnit1;			    /// handle to texture unit variable in fragment program
static GLint	h_vertex_scale;			/// handle to uniform variable in vertex program
static GLint	h_window_ratio;			/// handle to uniform variable in vertex program
static GLint	h_gradient_loc;			/// handle to uniform variable in fragment program
static GLint	h_color_mask;			/// handle to uniform variable in fragment program



// C A L L B A C K S ///////////////////////////////////////////////////


// _____________________________________________________
//|														|
//|	 display											|
//|_____________________________________________________|
///
///  Whenever OpenGL requires a screen refresh
///  it will call display() to draw the scene.
///  We specify that this is the correct function
///  to call with the glutDisplayFunc() function
///  during initialization

void display(void)
{ 
	safe_glUseProgram(h_program);
	
	safe_glUniform1i(h_texUnit0, 0);
	safe_glUniform1i(h_texUnit1, 1);
	safe_glUniform1f(h_vertex_scale, g_obj_scale);
	safe_glUniform1f(h_window_ratio, (float) g_width / g_height);
	safe_glUniform1f(h_gradient_loc, g_gradient_loc);
	safe_glUniform3f(h_color_mask, g_red_mask,g_green_mask,g_blue_mask, 1.);

	//clear the screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//draw two triangles
	glBegin(GL_TRIANGLES);

	// first triangle
	safe_glVertexAttrib2f(h_vTexCoord,0.0, 0.0);
	glVertex2f(-0.5f, -0.5f);

	safe_glVertexAttrib2f(h_vTexCoord,0.0, 1.0);
	glVertex2f(-0.5f,  0.5f);

	safe_glVertexAttrib2f(h_vTexCoord,1.0, 0.0);
	glVertex2f( 0.5f, -0.5f);

	// second triangle
	safe_glVertexAttrib2f(h_vTexCoord,1.0, 0.0);
	glVertex2f( 0.5f, -0.5f);

	safe_glVertexAttrib2f(h_vTexCoord,0.0, 1.0);
	glVertex2f(-0.5f,  0.5f);

	safe_glVertexAttrib2f(h_vTexCoord,1.0, 1.0);
	glVertex2f( 0.5f,  0.5f);

	glEnd();
	glutSwapBuffers();

	// check for errors
	
	GLenum errCode = glGetError();
	if (errCode != GL_NO_ERROR){
		const GLubyte *errString;
		errString=gluErrorString(errCode);
		printf("error: %s\n", errString);
	}

}


// _____________________________________________________
//|														|
//|	 reshape											|
//|_____________________________________________________|
///
///  Whenever a window is resized, a "resize" event is
///  generated and glut is told to call this reshape
///  callback function to handle it appropriately.

void reshape(int w, int h)
{
	g_width = w;
	g_height = h;
	glViewport(0, 0, w, h);

	printf("Size of window is now  %d x %d\n", w, h);
	glutPostRedisplay();
}


// _____________________________________________________
//|														|
//|	 mouse												|
//|_____________________________________________________|
///
///  Whenever a mouse button is clicked, a "mouse" event
///  is generated and this mouse callback function is
///  called to handle the user input.

void mouse(int button, int state, int x, int y) 
{
	if (button == GLUT_LEFT_BUTTON)
	{
		if (state == GLUT_DOWN) 
		{           
			// left mouse button has been clicked
			g_left_clicked = true;
			g_lclick_x = x;
			g_lclick_y = g_height - y - 1;
		}
		else
		{
			// left mouse button has been released
			g_left_clicked = false;
			// to work around the user dragging the mouse far in one direction,
			// releasing the button and having to drag back the same amount.
			g_gradient_loc = (g_gradient_loc > 1.0) ? 1.0 : g_gradient_loc;
			g_gradient_loc = (g_gradient_loc < -1.0) ? -1.0 : g_gradient_loc;
		}
	}
	if (button == GLUT_RIGHT_BUTTON)
	{
		if (state == GLUT_DOWN) 
		{           
			// right mouse button has been clicked
			g_right_clicked = true;
			g_rclick_x = x;
			g_rclick_y = g_height - y - 1;
		}
		else
		{
			// right mouse button has been released
			g_right_clicked = false;
		}
	}
}


// _____________________________________________________
//|														|
//|	 motion											|
//|_____________________________________________________|
///
///  Whenever the mouse is moved while a button is pressed, 
///  a "mouse move" event is triggered and this callback is 
///  called to handle the event.

static void motion(int x, int y)
{
	const int newx = x;
	const int newy = g_height - y - 1;
	if (g_left_clicked)
	{
		float deltax = (newx - g_lclick_x) * 0.01;
		g_gradient_loc -= deltax;
		g_lclick_x = newx;
		g_lclick_y = newy;
	}
	if (g_right_clicked)
	{
		float deltax = (newx - g_rclick_x) * 0.02;
		g_obj_scale += deltax;

		g_rclick_x = newx;
		g_rclick_y = newy;
	}
	glutPostRedisplay();
}



void keyboard(unsigned char key, int x, int y)
{
	switch(key)
	{
		case 'h':
			printf(" ============== H E L P ==============\n\n");
			printf("h\t\thelp menu\n");
			printf("s\t\tsave screenshot\n");
			printf("drag right mouse to change square size\n");
			break;
		case 'q': exit(0);
		case 'f': //flip the gradient of the textures
			break;
			//the next three toggle colors
		case 'r':
			g_red_mask = (g_red_mask == 1.0) ? 0. : 1.;
			glutPostRedisplay();
			break;
		case 'g':
			g_green_mask = (g_green_mask == 1.0) ? 0. : 1.;
			glutPostRedisplay();
			break;
		case 'b':
			g_blue_mask = (g_blue_mask == 1.0) ? 0. : 1.;
			glutPostRedisplay();
			break;
		case 's':
			glFinish();
			WritePPMScreenshot(g_width, g_height, "out.ppm");
			break;
	}
}



// H E L P E R    F U N C T I O N S ////////////////////////////////////


void initGlutState(int argc, char **argv)
{
	glutInit(&argc,argv);					// initialize Glut based on cmd-line args
	glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);	//  RGBA pixel channels and double buffering
	glutInitWindowSize(g_width, g_height);	// create a window
	glutCreateWindow("CS 175: Hello World");	// title the window

	glutDisplayFunc(display);					// display rendering callback
	glutReshapeFunc(reshape);					// window reshape callback
	glutMotionFunc(motion);				// mouse movement callback
	glutMouseFunc(mouse);						// mouse click callback
	glutKeyboardFunc(keyboard);
}

void InitGLState()
{
	glClearColor(0.,0.2 ,0.,0.);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);

	// set up texture
	safe_glActiveTexture(GL_TEXTURE0);
	int twidth, theight;
	packed_pixel_t *pixdata = ppmread("reachup.ppm", &twidth, &theight);
	assert(pixdata);
	glGenTextures(1, &h_texture);
	glBindTexture(GL_TEXTURE_2D, h_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, twidth, theight, 0, GL_RGB, GL_UNSIGNED_BYTE, pixdata);
	free(pixdata);

	//second texture
	safe_glActiveTexture(GL_TEXTURE1);
	pixdata = ppmread("mountain.ppm", &twidth, &theight);
	glGenTextures(1, &h_texture2);
	glBindTexture(GL_TEXTURE_2D, h_texture2);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, twidth, theight, 0, GL_RGB, GL_UNSIGNED_BYTE, pixdata);
	free(pixdata);
}



// M A I N /////////////////////////////////////////////////////////////

// _____________________________________________________
//|														|
//|	 main												|
//|_____________________________________________________|
///
///  The main entry-point for the HelloWorld example 
///  application.

int main(int argc, char **argv) 
{  
	
	initGlutState(argc,argv);
	glewInit();								// load	the	OpenGL extensions

	// Check that	our	graphics card and driver support the necessary extensions
	if (!GLEW_VERSION_2_0)
	{
		// Check that	our	graphics card and driver support the necessary extensions
		if (glewGetExtension("GL_ARB_fragment_shader")!= GL_TRUE || glewGetExtension("GL_ARB_vertex_shader")!= GL_TRUE ||
			glewGetExtension("GL_ARB_shader_objects") != GL_TRUE ||	glewGetExtension("GL_ARB_shading_language_100") != GL_TRUE)
		{
			std::cerr << "Error: card/driver does not support OpenGL Shading Language\n";
			assert(0);
		}
	}

	InitGLState();

	if (!ShaderInit("./shaders/asst1.vshader", "./shaders/asst1.fshader", &h_program))
	{
		fprintf(stderr, "Error: could not build the shaders\n");
		assert(0);
	}

	// grab handles to the shader variables by name
	h_vTexCoord = safe_glGetAttribLocation(h_program, "vTexCoord");
	h_texUnit0 = safe_glGetUniformLocation(h_program, "texUnit0");
	h_texUnit1 = safe_glGetUniformLocation(h_program, "texUnit1");
	h_vertex_scale = safe_glGetUniformLocation(h_program, "VertexScale");
	h_window_ratio = safe_glGetUniformLocation(h_program, "WindowRatio");
	h_gradient_loc = safe_glGetUniformLocation(h_program, "GradientLoc");
	h_color_mask = safe_glGetUniformLocation(h_program, "ColorMask");

	// Pass control to glut to handle	the	main program loop
	glutMainLoop();

	return 0;
}


