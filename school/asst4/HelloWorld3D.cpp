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
#include <cstddef>
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
#include "ppm.h"
#include "matrix4.h"
#include "shader.h"
#include "object3d.h"
#include "shape.h"
#include "floor.h"
#include "cube.h"
#include "skycam.h"
#include "shaderstate.h"
#include "rbt.h"
#include "quaternion.h"
#include "arcball.h"
#include "robot.h"



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
static int active_shader = 0;
static bool pickMode = false;

static vector <ShaderState> SState(2);													// initializes a vector with 2 ShaderState


static const char * const shader_file[2][2] = {{"./shaders/basic.vshader", "./shaders/diffuse.fshader"}, {"./shaders/basic.vshader", "./shaders/solid.fshader"}};



// --------- Scene

static const Vector3 Light(0, 3.0, 14.0);													// the light direction
static vector <Shape*> visibleObjects;													// the objects that are drawn
static vector <Object3D*> mutableObjects;													// the objects that can be manipulated
static Arcball* arcball;
static Object3D* currentMut;
static Object3D* currentEye;
static int eyePoseIndex;																	// index of current camera
static int mutableIndex;																	// index of current manipulated object
static Rbt auxilaryFrame;																// auxilary frame for current mutable
Matrix4 projmat = Matrix4::makeProjection(frust_fovy, window_width / static_cast <double> (window_height), frust_near, frust_far);


static void changeMutable(int indexDelta) {
	mutableIndex += indexDelta;
	int mutLen = mutableObjects.size();
	if (mutableIndex < 0) mutableIndex += mutLen;
	else if (mutableIndex >= mutLen) mutableIndex -= mutLen;
	currentMut = mutableObjects[mutableIndex];
	cout << "Manipulating " << currentMut->name_ << endl;
	cout << "Using " << currentEye->name_ << " as camera."<< endl;
}

static void changeMutable(Object3D* newMut) {
	currentMut = newMut;
	mutableIndex = -1;
	cout << "Manipulating " << currentMut->name_ << endl;
	cout << "Using " << currentEye->name_ << " as camera."<< endl;
}

static void changeCamera(int indexDelta) {
	eyePoseIndex += indexDelta;
	int mutLen = mutableObjects.size();
	if (eyePoseIndex < 0) eyePoseIndex += mutLen;
	else if (eyePoseIndex >= mutLen) eyePoseIndex -= mutLen;
	currentEye = mutableObjects[eyePoseIndex];
	cout << "Manipulating " << currentMut->name_ << endl;
	cout << "Using " << currentEye->name_ << " as camera."<< endl;
}
//populates 3d environment
static void initShapes() {


	Floor* floor = new Floor(Vector4(0.1, 0.95, 0.1, 1.0), floor_y, floor_size);
	Robot* redCube = new Robot("Red Robot", Rbt(Vector3(-2.0, 0.0, 0.0)), Vector4(1, 0, 0, 1));
	Robot* blueCube = new Robot("Blue Robot", Rbt(Vector3(2.0, 0.0, 0.0)), Vector4(0, 0, 1, 1));
	SkyCam* skyCam = new SkyCam(Rbt(Vector3(0.0, 0.5, 7.0)));
	arcball = new Arcball(Vector4(1., 1., 1., 1.), skyCam, 1.5);
	visibleObjects.push_back(floor);
	visibleObjects.push_back(redCube);
	visibleObjects.push_back(blueCube);
	visibleObjects.push_back(arcball);

	mutableObjects.push_back(skyCam);
	mutableObjects.push_back(redCube);
	mutableObjects.push_back(blueCube);

	mutableIndex = 0;
	eyePoseIndex = 0;
	currentMut = mutableObjects[mutableIndex];
	currentEye = mutableObjects[eyePoseIndex];
}

static void calcAuxFrame( ){
	Rbt objectFrame = currentMut->getFrame();
	if (currentMut->usingWorldSky()) {
			//uses axes of skycam, origin of world
			auxilaryFrame = Rbt(objectFrame.getRotation());
	} else {
		Rbt eyeFrame = currentEye->getFrame();
		//axes of current eyePose, origin of object
		auxilaryFrame = Rbt(objectFrame.getTranslation(), eyeFrame.getRotation());
	}
}

static void drawStuff()
{

	Rbt eyePose = currentEye->getFrame();
	Rbt eyePoseInverse = eyePose.getInverse();
	projmat = Matrix4::makeProjection(frust_fovy, window_width / static_cast <double> (window_height), frust_near, frust_far);
	GLfloat glmatrix[16];
	projmat.writeToColumnMajorMatrix(glmatrix);
	safe_glUniformMatrix4fv(SState[active_shader].h_uProjMatrix_, glmatrix);				// build & send proj. matrix to vshader
	
	const Vector3 lightE = eyePoseInverse * Light;													// light direction in eye coordinates
	safe_glUniform3f(SState[active_shader].h_uLightE_, lightE[0], lightE[1], lightE[2]);

	// draw objects
	for (size_t i = 0; i < visibleObjects.size(); ++i)
    {
		Object3D* temp = visibleObjects[i];
		//draw hack image if in pick mode
		if (pickMode) temp->drawId(SState[1], eyePoseInverse);

		else temp->draw(SState[active_shader], eyePoseInverse);

	}
	glFlush();
}
// ???
static void easterEgg() {
			double red = ((double) rand())/ RAND_MAX;
			double green = ((double) rand())/ RAND_MAX;
			double blue = ((double) rand())/ RAND_MAX;
			double size = 1.5 * ((double) rand())/ RAND_MAX + .5;
			double x =  16 * (((double) rand())/ RAND_MAX) - 8;
			double y =  4 * (((double) rand())/ RAND_MAX) - 1;
			double z =  16 * (((double) rand())/ RAND_MAX) - 8;
			double thetax = (((double) rand())/ RAND_MAX) * 180;
			double thetay = (((double) rand())/ RAND_MAX) * 180;
			double thetaz = (((double) rand())/ RAND_MAX) * 180;
			Rbt newFrame = Rbt(Vector3(x,y,z), Quaternion::makeXRotation(thetax) *
				Quaternion::makeYRotation(thetay) * Quaternion::makeZRotation(thetaz));
			Robot* newCube = new Robot("Easter Robot", newFrame, Vector4(red, green, blue, 1.0));
			visibleObjects.push_back(newCube);
			mutableObjects.push_back(newCube);
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
	if (currentMut->getId() != currentEye->getId() && !currentMut->editableRemotely()) return;
    const double dx = x - mouse_click_x;
    const double dy = window_height - y - 1 - mouse_click_y;
	if (!currentMut->usesDefaultTransformBehavior()) currentMut->interpretMouseMotion(dx, dy);
	else {
		calcAuxFrame();
		Rbt transform;

		if (mouse_lclick_button && !mouse_rclick_button)						// if the left mouse-button is down
		{
			//rotation
			if (currentMut->getId() == currentEye->getId() && !(currentMut->usesArcball() && currentMut->usingWorldSky())) {
				transform = Rbt(Quaternion::makeXRotation(dy) * Quaternion::makeYRotation(-dx));
			}
			else {
			transform = arcball->updateRotation(x, window_height - y - 1);
			}
		}
		if (mouse_rclick_button && !mouse_lclick_button)
		{
			//translation, x and y
			transform = Rbt(Vector3(dx, dy, 0) * 0.01);
		}
		if (mouse_mclick_button || (mouse_lclick_button && mouse_rclick_button))
		{
			//translation, z
			transform = Rbt(Vector3(0, 0, -dy) * 0.01);
		}
		//apply transformation WRT aux frame
		currentMut->preApplyTransformation(auxilaryFrame * transform * auxilaryFrame.getInverse());
		//update aux frame after changes	
	}
	if (mouse_click_down) glutPostRedisplay();												// we always redraw if we changed the scene
    mouse_click_x = x;
    mouse_click_y = window_height - y - 1;
}


//gets the object clicked on using hack image
static Object3D * getClickedObject(int x, int y) {
	float * point = new float[0];
	glReadPixels(x,y, 1, 1, GL_RED, GL_FLOAT, point);
	int id = Object3D::convertColorToId(point[0]);
	cout<< id <<endl;
	for (size_t i = 0; i < visibleObjects.size(); ++i)
    {
		Object3D* temp = visibleObjects[i];
		Object3D* clicked = temp->findId(id);
		if (clicked != NULL) return clicked;
    }
	//otherwise, skycam
	return mutableObjects[0];
}

static void mouse(const int button, const int state, const int x, const int y)
{
    mouse_click_x = x;
    mouse_click_y = window_height - y - 1;
    mouse_click_down = state == GLUT_DOWN;													// conversion from GLUT window-coordinate-system to OpenGL window-coordinate-system
    if (button == GLUT_LEFT_BUTTON && mouse_click_down) mouse_lclick_button = true;
    if (button == GLUT_RIGHT_BUTTON && mouse_click_down) mouse_rclick_button = true;
    if (button == GLUT_MIDDLE_BUTTON && mouse_click_down) mouse_mclick_button = true;
    if (button == GLUT_LEFT_BUTTON && !mouse_click_down) mouse_lclick_button = false;
    if (button == GLUT_RIGHT_BUTTON && !mouse_click_down) mouse_rclick_button = false;
    if (button == GLUT_MIDDLE_BUTTON && !mouse_click_down) mouse_mclick_button = false;
	//finds the clicked on object if in pick mode
	if (pickMode) {
		if (mouse_lclick_button && !mouse_rclick_button ) {
			pickMode = false;
			active_shader = 0;
			glClearColor(128./255., 200./255., 255./255., 0.);
			changeMutable(getClickedObject(mouse_click_x, mouse_click_y));
			arcball->setObjectToRotate(currentMut);
		}
	}
	//starts arcball movement if necessary
	if(mouse_lclick_button && !mouse_rclick_button && currentMut->usesArcball()) {
			arcball->setScreenSpaceCircle(currentEye->getFrame(), projmat, frust_near, frust_fovy, window_width, window_height);
			arcball->startRotation(mouse_click_x, mouse_click_y);
	}
	glutPostRedisplay();
}
static void keyboard(const unsigned char key, const int x, const int y)
{
	bool redisplay = true;
    switch (key)
    {
	    case 27: exit(0);																	// ESC
        case 'h':
            std::cerr << " ============== H E L P ==============\n\n";
            std::cerr << "h\t\thelp menu\n";
            std::cerr << "s\t\tsave screenshot\n";
            std::cerr << "f\t\tToggle flat shading on/off.\n";
            std::cerr << "m\t\tToggle pre/post-multiply\n";
            std::cerr << "v\t\tToggle view\n";
            std::cerr << "drag left mouse to rotate\n";
            break;
        case 's':
        	glFlush();
        	WritePPMScreenshot(window_width, window_height, "out.ppm");
            break;
        /*case 'f':
            active_shader ^= 1;
            break;*/
        case 'm': //switch coord system
				currentMut->switchMode();
            break;
		case 'v': //switch view
			changeCamera(1);
			arcball->setObjectToRotate(currentMut);
			break;
		case 'V': //switch view backwards
			changeCamera(-1);
			arcball->setObjectToRotate(currentMut);
			break;
		case 'o': //switch current mutable
			changeMutable(1);
			arcball->setObjectToRotate(currentMut);
			break;
		case 'O': //switch current mutable backwards
			changeMutable(-1);
			arcball->setObjectToRotate(currentMut);
			break;
		case 'p': //switch into pick mode, draw hack image to search for object
			pickMode = true;
			redisplay = false;
			glClearColor(1., 1., 1., 0.);
			active_shader = 1;
			safe_glUseProgram(SState[active_shader].h_program_);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);										// clear framebuffer color&depth
			drawStuff();
			//GLenum errCode = glGetError();													// check for errors
		    //if (errCode != GL_NO_ERROR) std::cerr << "Error: " << gluErrorString(errCode) << std::endl;
			break;
		case '?': //???
			easterEgg();
			break;
    }
    if (redisplay) glutPostRedisplay();
}
static void initGlutState(int argc, char * argv[])
{
  glutInit(&argc, argv);																	// initialize Glut based on cmd-line args
  glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);									//  RGBA pixel channels and double buffering
  glutInitWindowSize(window_width, window_height);											// create a window
  glutCreateWindow("Assignment 2");															// title the window
  
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
    }
    //safe_glUseProgram(h_program[0]);														// we can set the program (vertex/fragment shader pair) to use here, but will do so in "display()" instead
    
    glutMainLoop();																			// Pass control to glut to handle	the	main program loop

    return 0;
}