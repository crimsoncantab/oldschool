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
#include <string>
#include <list>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include "shader.h"
#include "object3d.h"
#include "shape.h"
#include "floor.h"
#include "skycam.h"
#include "shaderstate.h"
#include "meshcube.h"
#include "arcball.h"



// ----------- GL stuff
using namespace std;

static const float frust_fovy = 60.0;														// 60 degree field of view in y direction
static const float frust_near = -0.1;														// near plane
static const float frust_far = -50.0;														// far plane
static const float floor_y = -2.0;															// y coordinate of the floor
static const float floor_size = 10.0;														// half the floor length

static int window_width = 512;
static int window_height = 512;
static bool mouse_click_down = false;														// is the mouse button pressed
static bool mouse_lclick_button, mouse_rclick_button, mouse_mclick_button;
static int mouse_click_x, mouse_click_y;													// coordinates for mouse click event
static int active_shader = 1;
static int animSpeed = 4;
static bool worldSkyCoordSystem = true;													//whether to use sky-sky or sky-world coord system

static vector <ShaderState> SState(2);													// initializes a vector with 2 ShaderState


static const char * const shader_file[2][2] = {{"./shaders/basic.vshader", "./shaders/diffuse.fshader"}, {"./shaders/basic.vshader", "./shaders/specular.fshader"}};



// --------- Scene

static const Vector3 Light(10.0, 3.0, 25.0);													// the light direction
static vector <Shape*> visibleObjects;													// the objects that are drawn
static vector <Object3D*> mutableObjects;													// the objects that can be manipulated
static Arcball* arcball;
static SkyCam* skyCam;
static MeshCube* meshCube;
static int currentEyePose;																	// index of current camera
static int currentMutable;																	// index of current manipulated object
static Rbt auxilaryFrame;																// auxilary frame for current mutable
Matrix4 projmat = Matrix4::makeProjection(frust_fovy, window_width / static_cast <double> (window_height), frust_near, frust_far);

//populates 3d environment
static void initShapes() {


	Floor* floor = new Floor(Vector4(0.1, 0.95, 0.1, 1.0), floor_y, floor_size);
	Mesh m;
	m.load("./cube.mesh");
	meshCube = new MeshCube("Meshy Cube", Rbt(Vector3()),Vector4(.5,.2,.8, 1.),m);
	//Cube* redCube = new Cube("Cube 1", Rbt(Vector3(-2.0, 0.0, 0.0)), Vector4(1, 0, 0, 1), 1);
	//Cube* blueCube = new Cube("Cube 2", Rbt(Vector3(2.0, 0.0, 0.0)), Vector4(0, 0, 1, 1), 1);
	skyCam = new SkyCam(Rbt(Vector3(0.0, 0, 5.0)));
	arcball = new Arcball(Vector4(1., 1., 1., 1.), skyCam, 1.5, worldSkyCoordSystem);
	visibleObjects.push_back(floor);
	visibleObjects.push_back(meshCube);
	//visibleObjects.push_back(redCube);
	//visibleObjects.push_back(blueCube);
	//visibleObjects.push_back(arcball);

	mutableObjects.push_back(skyCam);
	//mutableObjects.push_back(redCube);
	//mutableObjects.push_back(blueCube);

	currentMutable = 0;
	currentEyePose = 0;
}

static void calcAuxFrame( ){
	Rbt objectFrame = mutableObjects[currentMutable]->getFrame();
	if (currentMutable == 0) { //using skycam
		if (worldSkyCoordSystem) {
			//uses axes of skycam, origin of world
			auxilaryFrame = Rbt(objectFrame.getRotation());
		} else {
			//uses own frame
			auxilaryFrame = objectFrame;
		}
	} else {
		Rbt eyeFrame = mutableObjects[currentEyePose]->getFrame();
		//axes of current eyePose, origin of object
		auxilaryFrame = Rbt(objectFrame.getTranslation(), eyeFrame.getRotation());
	}
}

static void drawStuff()
{

	Rbt eyePose = skyCam->getFrame();
	Rbt eyePoseInverse = eyePose.getInverse();
	projmat = Matrix4::makeProjection(frust_fovy, window_width / static_cast <double> (window_height), frust_near, frust_far);
	GLfloat glmatrix[16];
	projmat.writeToColumnMajorMatrix(glmatrix);
	safe_glUniformMatrix4fv(SState[active_shader].h_uProjMatrix_, glmatrix);				// build & send proj. matrix to vshader
	
	const Vector3 lightE = eyePoseInverse * Light;													// light direction in eye coordinates
	safe_glUniform3f(SState[active_shader].h_uLightE_, lightE[0], lightE[1], lightE[2]);

	// draw objects
	for (std::size_t i = 0; i < visibleObjects.size(); ++i)
    {
		Object3D* temp = visibleObjects[i];
		temp->draw(SState[active_shader], eyePoseInverse);
    }																	
}
static void display()
 {
    safe_glUseProgram(SState[active_shader].h_program_);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);										// clear framebuffer color&depth

    drawStuff();

    glutSwapBuffers();																		// show the back buffer (where we rendered stuff)

    const GLenum errCode = glGetError();													// check for errors
    if (errCode != GL_NO_ERROR) std::cerr << "Error: " << gluErrorString(errCode) << std::endl;
}
static void animateTimerCallback(int value) {
	value += animSpeed;
	value = value % 360;
	meshCube->scalify(value);
	glutPostRedisplay();
	glutTimerFunc(60, animateTimerCallback, value);
}

static void reshape(const int w, const int h)
{
    window_width = w;
    window_height = h;
    glViewport(0, 0, w, h);
    std::cerr << "Size of window is now " << w << "x" << h << std::endl;
    glutPostRedisplay();
}


static void motion(const int x, const int y)
{
	calcAuxFrame();
    const double dx = x - mouse_click_x;
    const double dy = window_height - y - 1 - mouse_click_y;
	Rbt transform;

    if (mouse_click_down && mouse_lclick_button && !mouse_rclick_button)						// if the left mouse-button is down
    {
		//rotation
		if (currentEyePose == currentMutable && ((currentMutable == 0 && worldSkyCoordSystem == false) || (currentMutable != 0))) {
			transform = Rbt(Quaternion::makeXRotation(dy) * Quaternion::makeYRotation(-dx));
		}
		else {
		transform = arcball->updateRotation(x, window_height - y - 1);
		}
    }
    if (mouse_click_down && mouse_rclick_button && !mouse_lclick_button)
    {
		//translation, x and y
        transform = Rbt(Vector3(dx, dy, 0) * 0.01);
    }
    if (mouse_click_down && (mouse_mclick_button || (mouse_lclick_button && mouse_rclick_button)))
    {
		//translation, z
        transform = Rbt(Vector3(0, 0, -dy) * 0.01);
    }
	//apply transformation WRT aux frame
	mutableObjects[currentMutable]->preApplyTransformation(auxilaryFrame * transform * auxilaryFrame.getInverse());
	//update aux frame after changes
	if (mouse_click_down) glutPostRedisplay();												// we always redraw if we changed the scene
    
    mouse_click_x = x;
    mouse_click_y = window_height - y - 1;
}


static void mouse(const int button, const int state, const int x, const int y)
{
    mouse_click_x = x;
    mouse_click_y = window_height - y - 1;													// conversion from GLUT window-coordinate-system to OpenGL window-coordinate-system
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) mouse_lclick_button = true;
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) mouse_rclick_button = true;
    if (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN) mouse_mclick_button = true;
    if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) mouse_lclick_button = false;
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP) mouse_rclick_button = false;
    if (button == GLUT_MIDDLE_BUTTON && state == GLUT_UP) mouse_mclick_button = false;
    mouse_click_down = state == GLUT_DOWN;

	if(mouse_click_down && mouse_lclick_button && !mouse_rclick_button) {
		arcball->setScreenSpaceCircle(mutableObjects[currentEyePose]->getFrame(), projmat, frust_near, frust_fovy, window_width, window_height);
		arcball->startRotation(mouse_click_x, mouse_click_y);
	}
}
static void keyboard(const unsigned char key, const int x, const int y)
{
    switch (key)
    {
	    case 27: exit(0);																	// ESC
        case 'h':
            std::cerr << " ============== H E L P ==============\n\n";
            std::cerr << "h\t\tHelp menu\n";
            std::cerr << "s\t\tSmooth shading\n";
            std::cerr << "f\t\tToggle shading program.\n";
            std::cerr << "<\t\tDecrease animation speed\n";
            std::cerr << ">\t\tIncrease animation speed\n";
            std::cerr << "-\t\tDecrease subdivisions\n";
            std::cerr << "+\t\tIncrease subdivisions\n";
            std::cerr << "Drag left mouse to rotate\n";
            break;
        case 's':
        	//glFlush();
        	//WritePPMScreenshot(window_width, window_height, "out.ppm");
            meshCube->changeShading();
			break;
        case 'f':
            active_shader ^= 1;
            break;
		case '=':
		case '+':
			meshCube->increaseSubdiv();
			break;
		case '-':
		case '_':
			meshCube->decreaseSubdiv();
			break;
		case ',':
		case '<':
			if (animSpeed > 1) animSpeed /= 2;
			break;
		case '.':
		case '>':
			if (animSpeed < 32) animSpeed *= 2;
			break;
    }
    glutPostRedisplay();
}
static void initGlutState(int argc, char * argv[])
{
  glutInit(&argc, argv);																	// initialize Glut based on cmd-line args
  glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);									//  RGBA pixel channels and double buffering
  glutInitWindowSize(window_width, window_height);											// create a window
  glutCreateWindow("Assignment 7");															// title the window
  
  glutDisplayFunc(display);																	// display rendering callback
  glutReshapeFunc(reshape);																	// window reshape callback
  glutMotionFunc(motion);																	// mouse movement callback
  glutMouseFunc(mouse);																		// mouse click callback
  glutKeyboardFunc(keyboard);
}

static void InitGLState()
{
  glClearColor(128./255., 200./255., 255./255., 0.);
  glClearDepth(0.);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glCullFace(GL_BACK);
  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_GREATER);
  glReadBuffer(GL_BACK);
}
int main(int argc, char * argv[])
{
	initShapes();
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
        SState[i].h_uLightE_ = safe_glGetUniformLocation(SState[i].h_program_, "uLight");	// Retrieve handles to variables in program
        SState[i].h_uProjMatrix_ = safe_glGetUniformLocation(SState[i].h_program_, "uProjMatrix");
        SState[i].h_uModelViewMatrix_ = safe_glGetUniformLocation(SState[i].h_program_, "uModelViewMatrix");
        SState[i].h_uNormalMatrix_ = safe_glGetUniformLocation(SState[i].h_program_, "uNormalMatrix");
        SState[i].h_vColor_ = safe_glGetAttribLocation(SState[i].h_program_, "vColor");
    }													// we can set the program (vertex/fragment shader pair) to use here, but will do so in "display()" instead
    animateTimerCallback(0);
    glutMainLoop();																			// Pass control to glut to handle	the	main program loop

    return 0;
}