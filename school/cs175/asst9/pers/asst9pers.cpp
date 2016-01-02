////////////////////////////////////////////////////////////////////////
//
//	 Harvard University
//   CS175 : Computer Graphics
//   Professor Steven Gortler
//
//	 Formatted for 4-space tabs, wide lines
//
////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <GL/glew.h>
#ifdef __MAC__
#   include <GLUT/glut.h>
#else
#   include <GL/glut.h>
#endif
#include <vector>
#include <ctime>
#include <iostream>
#include <sstream>
#include "ppm.h"
#include "matrix4.h"
#include "shader.h"



// ----------- GL stuff

static const double frust_fovy = 60.0;														// 60 degree field of view in y direction
static const double frust_near = -0.1;														// near plane
static const double frust_far = -50.0;														// far plane
static const double floor_y = -2.0;															// y coordinate of the floor
static const double floor_size = 10.0;														// half the floor length


static int window_width = 512;
static int window_height = 512;
static bool mouse_click_down = false;														// is the mouse button pressed
static bool mouse_lclick_button, mouse_rclick_button, mouse_mclick_button;
static int mouse_click_x, mouse_click_y;													// coordinates for mouse click event
static int active_shader = 0;
static GLuint h_texture;																	// handle to texture (a OpenGL handle, not a shader handle)


static Matrix4 SkyPose = Matrix4(Matrix4::makeTranslation(Vector3(0.0, 0.0, 3.0)));
static Matrix4 ObjectFrame = Matrix4::makeTranslation(Vector3(0,0,0));


struct ShaderState
{
    GLuint h_program_;
    GLint h_uProjMatrix_;
    GLint h_uModelViewMatrix_;
    GLint h_vTexCoord_;
    GLint h_vTexUnit0_;
};

static std::vector <ShaderState> SState(2);													// initializes a vector with 2 ShaderState

static const char * const shader_file[2][2] =
{
    {"./shaders/persptex1.vshader", "./shaders/persptex1.fshader"},
    {"./shaders/persptex2.vshader", "./shaders/persptex2.fshader"}
};




static void drawStuff()
{
	GLfloat glmatrix[16];
	const Matrix4 projmat = Matrix4::makeProjection(frust_fovy, window_width / static_cast <double> (window_height), frust_near, frust_far);
	projmat.writeToColumnMajorMatrix(glmatrix);
	safe_glUniformMatrix4fv(SState[active_shader].h_uProjMatrix_, glmatrix);				// build & send proj. matrix to vshader
	
	const Matrix4 EyePose = SkyPose;
	const Matrix4 inveye = EyePose.getAffineInverse();  
	const Matrix4 MVM = inveye * ObjectFrame;
	MVM.writeToColumnMajorMatrix(glmatrix);													// send MVM
	safe_glUniformMatrix4fv(SState[active_shader].h_uModelViewMatrix_, glmatrix);


	// draw two triangles
    glBegin(GL_TRIANGLES);

    // first triangle
    safe_glVertexAttrib2f(SState[active_shader].h_vTexCoord_, 0.0, 1.0);
    glVertex4f(-1.5f,  1.5f, 0.0, 1.0);

    safe_glVertexAttrib2f(SState[active_shader].h_vTexCoord_, 0.0, 0.0);
    glVertex4f(-1.5f, -1.5f, 0.0, 1.0);

    safe_glVertexAttrib2f(SState[active_shader].h_vTexCoord_, 1.0, 0.0);
    glVertex4f( 1.5f, -1.5f, 0.0, 1.0);

    // second triangle
    safe_glVertexAttrib2f(SState[active_shader].h_vTexCoord_, 0.0, 1.0);
    glVertex4f(-1.5f,  1.5f, 0.0, 1.0);

    safe_glVertexAttrib2f(SState[active_shader].h_vTexCoord_, 1.0, 0.0);
    glVertex4f( 1.5f, -1.5f, 0.0, 1.0);

    safe_glVertexAttrib2f(SState[active_shader].h_vTexCoord_, 1.0, 1.0);
    glVertex4f( 1.5f,  1.5f, 0.0, 1.0);
  
    glEnd();
}



static void display()
{
    safe_glUseProgram(SState[active_shader].h_program_);
	safe_glUniform1i(SState[active_shader].h_vTexUnit0_, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);										// clear framebuffer color&depth

    drawStuff();

    glutSwapBuffers();																		// show the back buffer (where we rendered stuff)

    const GLenum errCode = glGetError();													// check for errors
    if (errCode != GL_NO_ERROR) std::cerr << "Error: " << gluErrorString(errCode) << std::endl;
}

static void reshape(const int w, const int h)
{
    window_width = w;
    window_height = h;
    glViewport(0, 0, w, h);
    std::cerr << "Size of window is now " << w << "x" << h << std::endl;
    glutPostRedisplay();
}



static Matrix4 do_Q_to_O_wrt_A(const Matrix4& O, const Matrix4& Q, const Matrix4& A)
{
    return A * Q * A.getAffineInverse() * O;
}

static Matrix4 getQMatrix(const double dx, const double dy)
{
    Matrix4 Q = mouse_lclick_button && !mouse_rclick_button ? Matrix4::makeXRotation(-dy) * Matrix4::makeYRotation(dx) : (
                mouse_rclick_button && !mouse_lclick_button ? Matrix4::makeTranslation(Vector3(dx, dy, 0) * 0.01) : 
                                                              Matrix4::makeTranslation(Vector3(0, 0, -dy) * 0.01));
    return Q;
}
static Matrix4 getAMatrix()								{ return ObjectFrame.getTranslation() * SkyPose.getRotation(); }

static void motion(const int x, const int y)
{
    if (!mouse_click_down) return;

    const double dx = x - mouse_click_x;
    const double dy = window_height - y - 1 - mouse_click_y;

    const Matrix4 Q = getQMatrix(dx, dy);													// the "action" matrix
    const Matrix4 A = getAMatrix();															// the matrix for the auxiliary frame (the w.r.t.)

    ObjectFrame = do_Q_to_O_wrt_A(ObjectFrame, Q, A);
    
    mouse_click_x = x;
    mouse_click_y = window_height - y - 1;
    glutPostRedisplay();																	// we always redraw if we changed the scene
}


static void mouse(const int button, const int state, const int x, const int y)
{
    mouse_click_x = x;
    mouse_click_y = window_height - y - 1;													// conversion from GLUT window-coordinate-system to OpenGL window-coordinate-system
    mouse_click_down = state == GLUT_DOWN;
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) mouse_lclick_button = true;
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) mouse_rclick_button = true;
    if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN) mouse_mclick_button = true;
    if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) mouse_lclick_button = false;
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP) mouse_rclick_button = false;
    if (button == GLUT_MIDDLE_BUTTON && state == GLUT_UP) mouse_mclick_button = false;
}


static void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	    case 27:
	        exit(0);
	        break;
    	case '1':
    	    active_shader = 0;
            break;
        case '2':
            active_shader = 1;
            break;
    	case 'h':
    	    std::printf(" ============== H E L P ==============\n\n");
            std::printf("h\t\thelp menu\n");
            std::printf("s\t\tsave screenshot\n");
            std::printf("drag left mouse to rotate square\n");
            break;
        case 's':
        	glFlush();
        	WritePPMScreenshot(window_width, window_height, "out.ppm");
            break;
	}
    glutPostRedisplay();
}





static void initGlutState(int argc, char * argv[])
{
    glutInit(&argc, argv);																	// initialize Glut based on cmd-line args
    glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);									//  RGBA pixel channels and double buffering
    glutInitWindowSize(window_width, window_height);										// create a window
    glutCreateWindow("Assignment 9");														// title the window
  
    glutDisplayFunc(display);																// display rendering callback
    glutReshapeFunc(reshape);																// window reshape callback
    glutMotionFunc(motion);																	// mouse movement callback
    glutMouseFunc(mouse);																	// mouse click callback
    glutKeyboardFunc(keyboard);
}

static void InitGLState()
{
    glClearColor(100./255., 150./255., 100./255., 0.);
    glClearDepth(0.);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    glReadBuffer(GL_BACK);

    // set up texture
    glActiveTexture(GL_TEXTURE0);
    int twidth, theight;
    packed_pixel_t * pixdata = ppmread("reachup.ppm", &twidth, &theight);
    glGenTextures(1, &h_texture);
    glBindTexture(GL_TEXTURE_2D, h_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, twidth, theight, 0, GL_RGB, GL_UNSIGNED_BYTE, pixdata);
    free(pixdata);																			// C-style
}



int main(int argc, char **argv) 
{  
    initGlutState(argc,argv);  

    glewInit();																				// load	the	OpenGL extensions
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
    InitGLState();																			// this is our own ftion for setting some GL state
  
    for (std::size_t i = 0; i < SState.size(); ++i)
    {
        const int shadeRet = ShaderInit(shader_file[i][0], shader_file[i][1], &SState[i].h_program_);
        if (!shadeRet)
        {
            std::cerr << "Error: could not build the shaders " << shader_file[i][0] << ", and " << shader_file[i][1] << ". Exiting...\n";
            assert(0);
        }        
        SState[i].h_uProjMatrix_ = safe_glGetUniformLocation(SState[i].h_program_, "uProjMatrix");
        SState[i].h_uModelViewMatrix_ = safe_glGetUniformLocation(SState[i].h_program_, "uModelViewMatrix");
        SState[i].h_vTexCoord_ = safe_glGetAttribLocation(SState[i].h_program_, "vTexCoord");
        SState[i].h_vTexUnit0_ = safe_glGetUniformLocation(SState[i].h_program_, "vTexUnit0");
    }
    
    glutMainLoop();																			// Pass control to glut to handle	the	main program loop

    return 0;
}

