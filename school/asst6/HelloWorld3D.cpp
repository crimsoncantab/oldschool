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
#include <iostream>
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
#include "keyframe.h"
#include "framestate.h"



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
//static int active_shader = 0;
static bool pickMode = false;

static vector <ShaderState> SState(2);													// initializes a vector with 2 ShaderState


static const char * const shader_file[2][2] = {{"./shaders/basic.vshader", "./shaders/diffuse.fshader"}, {"./shaders/basic.vshader", "./shaders/solid.fshader"}};



// --------- Scene

static const Vector3 Light(0, 3.0, 14.0);													// the light direction
static FrameState* frameState;
Matrix4 projmat = Matrix4::makeProjection(frust_fovy, window_width / static_cast <double> (window_height), frust_near, frust_far);
static list <KeyFrame*> keyFrames;
static list <KeyFrame*>::iterator curKeyFrame;
static const int max_mill_between_keyframes = 5000;
static int millisec_between_keyframes = 2000;												// Controls speed of playback.
static int millisec_per_frame = 1000/60;													// Draw about 60 frames in each second
static int cur_anim_index = 0;
static int anim_n_value = keyFrames.size()-3;




//populates 3d environment
static void initShapes() {

	Floor* floor = new Floor(Vector4(0.1, 0.95, 0.1, 1.0), floor_y, floor_size);
	Robot* redCube = new Robot("Red Robot", Rbt(Vector3(-2.0, 0.0, 0.0)), Vector4(1, 0, 0, 1));
	Robot* blueCube = new Robot("Blue Robot", Rbt(Vector3(2.0, 0.0, 0.0)), Vector4(0, 0, 1, 1));
	SkyCam* skyCam = new SkyCam(Rbt(Vector3(0.0, 0.5, 7.0)));
	Arcball* arcball = new Arcball(Vector4(1., 1., 1., 1.), skyCam, 1.5);
	vector <Object3D*> objects;
	objects.push_back(skyCam);
	objects.push_back(floor);
	objects.push_back(redCube);
	objects.push_back(blueCube);
	frameState = new FrameState(objects, arcball, Light);
	KeyFrame * copy = new KeyFrame(*frameState);
	keyFrames.push_back(copy);
	curKeyFrame = keyFrames.begin();
}


static void drawStuff()
{

	int activeShader = (pickMode) ? 1: 0;
	GLfloat glmatrix[16];
	projmat.writeToColumnMajorMatrix(glmatrix);
	safe_glUniformMatrix4fv(SState[activeShader].h_uProjMatrix_, glmatrix);				// build & send proj. matrix to vshader
	frameState->draw(SState[activeShader], pickMode);
	glFlush();
}
// ???
//static void easterEgg() {
//			double red = ((double) rand())/ RAND_MAX;
//			double green = ((double) rand())/ RAND_MAX;
//			double blue = ((double) rand())/ RAND_MAX;
//			double size = 1.5 * ((double) rand())/ RAND_MAX + .5;
//			double x =  16 * (((double) rand())/ RAND_MAX) - 8;
//			double y =  4 * (((double) rand())/ RAND_MAX) - 1;
//			double z =  16 * (((double) rand())/ RAND_MAX) - 8;
//			double thetax = (((double) rand())/ RAND_MAX) * 180;
//			double thetay = (((double) rand())/ RAND_MAX) * 180;
//			double thetaz = (((double) rand())/ RAND_MAX) * 180;
//			Rbt newFrame = Rbt(Vector3(x,y,z), Quaternion::makeXRotation(thetax) *
//				Quaternion::makeYRotation(thetay) * Quaternion::makeZRotation(thetaz));
//			Robot* newCube = new Robot("Easter Robot", newFrame, Vector4(red, green, blue, 1.0));
//			visibleObjects.push_back(newCube);
//			mutableObjects.push_back(newCube);
//}

static void display()
 {
    safe_glUseProgram(SState[(pickMode) ? 1: 0].h_program_);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);										// clear framebuffer color&depth

    drawStuff();

    glutSwapBuffers();																		// show the back buffer (where we rendered stuff)

    const GLenum errCode = glGetError();													// check for errors
    if (errCode != GL_NO_ERROR) cerr << "Error: " << gluErrorString(errCode) << endl;
}
// Given t in the range [0, n], perform interpolation and draw the scene
// for the particular t. Returns true if we are at the end of the animation
// sequence, or false otherwise.
bool interpolateAndDisplay(float t) {
	if (t > 1) t = 1;
	curKeyFrame--;
	KeyFrame * prev = *curKeyFrame;
	curKeyFrame++;
	KeyFrame * f1 = *curKeyFrame;
	curKeyFrame++;
	KeyFrame * f2 = *curKeyFrame;
	curKeyFrame++;
	KeyFrame * post = *curKeyFrame;
	curKeyFrame--;
	curKeyFrame--;
	frameState->bezerpFrames(*f1, *f2, *prev, *post, t);
	glutPostRedisplay();
	if (t==1) return true;
	else return false;
}
// Interpret "value" as milliseconds into the animation
static void animateTimerCallback(int value) {
	float t = (float)value/(float)millisec_between_keyframes;
	if (!interpolateAndDisplay(t))
		glutTimerFunc(millisec_per_frame, animateTimerCallback, value + millisec_per_frame);
	else {
		cur_anim_index++;
		curKeyFrame++;
		if (cur_anim_index == anim_n_value) {
			//we're done
			frameState->stopAnim();
			glutPostRedisplay();
		} else {
			glutTimerFunc(millisec_per_frame, animateTimerCallback, 0);
		}
	}
}

static void reshape(const int w, const int h)
{
    window_width = w;
    window_height = h;
    glViewport(0, 0, w, h);
    cerr << "Size of window is now " << w << "x" << h << endl;
	projmat = Matrix4::makeProjection(frust_fovy, window_width / static_cast <double> (window_height), frust_near, frust_far);
    glutPostRedisplay();
}


static void motion(const int x, const int y)
{
	//if (currentMut->getId() != currentEye->getId() && !currentMut->editableRemotely()) return;
    const double dx = x - mouse_click_x;
    const double dy = window_height - y - 1 - mouse_click_y;
	mouse_click_x = x;
    mouse_click_y = window_height - y - 1;
	//frameState->motion(x, y, dx, dy);

	if (mouse_lclick_button && !mouse_rclick_button)						// if the left mouse-button is down
	{
		frameState->rotation(x, window_height - y - 1, dx, dy);
	}
	else if (mouse_rclick_button && !mouse_lclick_button)
	{
		//translation, x and y
		frameState->translation(dx, dy, false);
	}
	else if (mouse_mclick_button || (mouse_lclick_button && mouse_rclick_button))
	{
		//translation, z
		frameState->translation(dx, dy, true);
	}
	if (mouse_click_down) glutPostRedisplay();												// we always redraw if we changed the scene
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
	
	//starts arcball movement if necessary
	if(mouse_lclick_button && !mouse_rclick_button) {
		if (pickMode) {
			pickMode = false;
			glClearColor(128./255., 200./255., 255./255., 0.);
			frameState->changeMutable(mouse_click_x, mouse_click_y, 0);
			glutPostRedisplay();
		}
		if (frameState->usingArcball()) {
			frameState->setScreenSpaceCircle(projmat, frust_near, frust_fovy, window_width, window_height);
			frameState->startRotation(mouse_click_x, mouse_click_y);
		}
	}
}
static bool checkAnimation() {
	if (frameState->isAnimating()) {
		cout<<"Animation in progress."<<endl;
		return false;
	}
	return true;
}
static void keyboard(const unsigned char key, const int x, const int y)
{
	list<KeyFrame*> newKeyFrames;
    switch (key)
    {
	    case 27: exit(0);																	// ESC
        case 'h':
            cerr << " ============== H E L P ==============\n\n";
            cerr << "h\t\thelp menu\n";
            cerr << "s\t\tsave screenshot\n";
            cerr << "m\t\tToggle pre/post-multiply\n";
            cerr << "v/V\t\tToggle view\n";
            cerr << "o/O\t\tToggle object to manipulate\n";
            cerr << "p\t\tSwitch to pick mode.  Click on an object to manipulate.\n";
            cerr << "spc\t\tRevert current state to current key frame.\n";
            cerr << "u\t\tUpdate current key frame with current state.\n";
            cerr << ">/.\t\tGo to next key frame.\n";
            cerr << "</,\t\tGo to previous key frame.\n";
            cerr << "d\t\tDelete key frame, then go to previous key frame.\n";
            cerr << "n\t\tCreate new key frame and make it current key frame.\n";
            cerr << "w\t\tSave key frames.\n";
            cerr << "i\t\tLoad key frames.\n";
            cerr << "y\t\tIf the number of rames are > 4, animate.\n";
            cerr << "+/=\t\tIncrease animation speed.\n";
            cerr << "-/_\t\tDecrease animation speed.\n";
            cerr << "drag left mouse to rotate\n";
            cerr << "drag right mouse to translate\n";
            cerr << "drag middle/both mouse buttons to translate along z-axis\n";
            break;
        case 's':
        	glFlush();
			cout<<"Taking Screenshot."<<endl;
        	WritePPMScreenshot(window_width, window_height, "out.ppm");
            break;
        case 'm': //switch coord system
			frameState->switchMode();
            break;
		case 'v': //switch view
			frameState->changeCamera(1);
			break;
		case 'V': //switch view backwards
			frameState->changeCamera(-1);
			break;
		case 'o': //switch current mutable
			frameState->changeMutable(1);
			break;
		case 'O': //switch current mutable backwards
			frameState->changeMutable(-1);
			break;
		case 'p': //switch into pick mode, draw hack image to search for object
			if (!checkAnimation()) {break;}
			pickMode = true;
			glClearColor(1., 1., 1., 0.);
			safe_glUseProgram(SState[(pickMode)?1:0].h_program_);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);										// clear framebuffer color&depth
			drawStuff();
			cout << "In pick mode...";
			break;
		case ' ':
			if (!checkAnimation()) {break;}
			frameState->copyFrom(**curKeyFrame);
			cout<<"Updating current state with current key frame."<<endl;
			break;
		case 'u':
			if (!checkAnimation()) {break;}
			(*curKeyFrame)->copyFrom(*frameState);
			cout<<"Updating key frame."<<endl;
			break;
		case '>':
		case '.':
			if (!checkAnimation()) {break;}
			curKeyFrame++;
			if (curKeyFrame == keyFrames.end()) {curKeyFrame--;}
			else {
				frameState->copyFrom(**curKeyFrame);
				cout<<"Switching to next key frame."<<endl;
			}
			break;
		case '<':
		case ',':
			if (!checkAnimation()) {break;}
			if (curKeyFrame != keyFrames.begin()) {
				curKeyFrame--;
				frameState->copyFrom(**curKeyFrame); 
				cout<<"Switching to previous key frame."<<endl;
			}
			break;
		case 'd':
			if (!checkAnimation()) {break;}
			if (curKeyFrame != keyFrames.begin()) {
				curKeyFrame = keyFrames.erase(curKeyFrame);
				curKeyFrame--;
				frameState->copyFrom(**curKeyFrame);
				anim_n_value = keyFrames.size()-3;
				cout<<"Deleting key frame."<<endl;
			}
			break;
		case 'n':
			if (!checkAnimation()) {break;}
			curKeyFrame++;
			curKeyFrame = keyFrames.insert(curKeyFrame, new KeyFrame(*frameState));
			anim_n_value = keyFrames.size()-3;
			cout<<"Copying key frame."<<endl;
			break;
		case 'w':
			if (!checkAnimation()) {break;}
			cout<<"Saving key frames."<<endl;
			KeyFrame::saveKeyFrames(keyFrames);
			break;
		case 'i':
			if (!checkAnimation()) {break;}
			newKeyFrames = KeyFrame::loadKeyFrames(frameState);
			if (!newKeyFrames.empty()) {
				cout<<"Loading key frames."<<endl;
				keyFrames = newKeyFrames;
				curKeyFrame = keyFrames.begin();
				frameState->copyFrom(**curKeyFrame);
				anim_n_value = keyFrames.size()-3;
			} else cout<<"No key frames to load."<<endl;
			break;
		case 'y':
			if (!checkAnimation()) {break;}
			if (anim_n_value > 0) {
				cout<<"Animating..."<<endl;
				curKeyFrame = keyFrames.begin();
				curKeyFrame++;
				cur_anim_index = 0;
				frameState->startAnim();
				animateTimerCallback(0);
			}
			else {
				cout<<"At least 4 frames required to animate."<<endl;
			}
			break;
		case '+':
		case '=':
			if (millisec_between_keyframes > 1000) {
				millisec_between_keyframes -= 1000;
				cout<<"Ms between keyframes: "<<millisec_between_keyframes<<endl;
			}
			break;
		case '-':
		case '_':
			if (millisec_between_keyframes < max_mill_between_keyframes) {
				millisec_between_keyframes += 1000;
				cout<<"Ms between keyframes: "<<millisec_between_keyframes<<endl;
			}
			break;
		//case '?': //???
			//easterEgg();
			//break;
    }
    if (!pickMode) glutPostRedisplay();
}
static void initGlutState(int argc, char * argv[])
{
  glutInit(&argc, argv);																	// initialize Glut based on cmd-line args
  glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);									//  RGBA pixel channels and double buffering
  glutInitWindowSize(window_width, window_height);											// create a window
  glutCreateWindow("Assignment 6");															// title the window
  
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
	//test();
	initShapes();
    initGlutState(argc,argv);  

    glewInit();																				// load	the	OpenGL extensions
    if (!GLEW_VERSION_2_0)
    {
        // Check that	our	graphics card and driver support the necessary extensions
    	if (glewGetExtension("GL_ARB_fragment_shader")!= GL_TRUE || glewGetExtension("GL_ARB_vertex_shader")!= GL_TRUE ||
            glewGetExtension("GL_ARB_shader_objects") != GL_TRUE ||	glewGetExtension("GL_ARB_shading_language_100") != GL_TRUE)
      	{
            cerr << "Error: card/driver does not support OpenGL Shading Language\n";
            assert(0);
	}
    }
    InitGLState();																			// this is our own ftion for setting some GL state
  
    for (size_t i = 0; i < SState.size(); ++i)
    {
        const int shadeRet = ShaderInit(shader_file[i][0], shader_file[i][1], &SState[i].h_program_);
        if (!shadeRet)
        {
            cerr << "Error: could not build the shaders " << shader_file[i][0] << ", and " << shader_file[i][1] << ". Exiting...\n";
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