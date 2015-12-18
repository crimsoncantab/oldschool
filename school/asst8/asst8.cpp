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
#include <list>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include "ppm.h"
#include "matrix4.h"
#include "rbt.h"
#include "shader.h"
#include "arcball.h"
#include "mesh.h"

using namespace std;
// ----------- GL stuff

static const double frust_fovy = 60.0;														// 60 degree field of view in y direction
static const double frust_near = -0.1;														// near plane
static const double frust_far = -50.0;														// far plane
static const double floor_y = -2.0;															// y coordinate of the floor
static const double floor_size = 10.0;														// half the floor length

enum {SKY=0, OBJECT0=1};
enum {WORLD_SKY=0, SKY_SKY=1};

static int window_width = 512;
static int window_height = 512;
static bool mouse_lclick_button, mouse_rclick_button, mouse_mclick_button;
static int mouse_click_x, mouse_click_y;													// coordinates for mouse click event
static int active_shader = 0;
static int active_object = 0;
static int active_eye = SKY;
static int active_camera_frame = WORLD_SKY;

static bool draw_shells = false;


static bool display_arcball = false;
static double arcball_radius = 2.5;

static Mesh mesh;

static std::vector <Vector3> tips, vtips;
static vector<Mesh> shells;

static float fur_height = 0.21;
static const int num_shells = 32;
static float texture_coord = 1;
static float stiffness = .3;
static float time_step = 0.01;
static const int num_sim_idle = 15;
static const Vector3 gravity(0, -0.5, 0);
static float damping = 0.96;


GLuint	h_texture_fins;								// texture handles
GLuint	h_texture_shells;


struct ShaderState
{
    GLuint	h_program_;
    GLuint	h_texture_;
    GLint	h_texUnit0_;
    GLint	h_texUnit1_;

    GLint	h_uLight_;								/// handle to uniform variable for light vector
    GLint	h_uProjMatrix_;							/// handle to unifrom var for projection matrix
    GLint	h_uModelViewMatrix_;					/// handle to uniform var for model view matrices for Base
    GLint	h_uNormalMatrix_;						/// handle to uniform var for transforming normals for Base

    GLint	h_colorAmbient_;						/// handle to attribute variable for color of vertex
    GLint	h_colorDiffuse_;						/// handle to attribute variable for color of vertex
    GLint	h_vtexCoord_;							/// handle to uniform variable for light vector
    GLint	h_vTangent_;
    GLint	h_valpha_exponent_;
};

static std::vector <ShaderState> SState(2);													// initializes a vector with 2 ShaderState

static const char * const shader_file[3][2] =
{
    {"./shaders/shader.vert", "./shaders/shader.frag"},										// basic rendering
    {"./shaders/shader_shells.vert", "./shaders/shader_shells.frag"}						// shells
};



// --------- Scene

static const Vector3 Light(10.0, 5.5, 17.0);												// light position
struct Cube
{
    Rbt pose_;
    Vector3 color_;
    Cube()														{}
    Cube(const Rbt& p, const Vector3& c) : pose_(p), color_(c) 	{}
};

static Rbt skyPose_(Vector3(0.0, 0.5, 7.0));
static Cube cube_ = Cube(Rbt(Vector3(0.0, 0.0, 0.0)), Vector3(1, 0, 0));


static void init_tips()
{
    vtips.resize(mesh.getNumVertices(), Vector3(0));
    tips.resize(mesh.getNumVertices());

	for (int i = 0; i < mesh.getNumVertices(); i++) {
		tips[i] = cube_.pose_ * (mesh.getVertex(i).getPosition() + mesh.getVertex(i).getNormal() * fur_height);
	}
    // TODO: initialize tips to "at-rest" hair tips in world coordinates
}

static void sendModelViewNormalMatrix(const Matrix4& MVM)									// takes MVM and sends it (and normal matrix) to the vertex shader
{
    GLfloat glmatrix[16];
	MVM.writeToColumnMajorMatrix(glmatrix);													// send MVM
	safe_glUniformMatrix4fv(SState[active_shader].h_uModelViewMatrix_, glmatrix);
	
	MVM.getNormalMatrix().writeToColumnMajorMatrix(glmatrix);								// send normal matrix
	safe_glUniformMatrix4fv(SState[active_shader].h_uNormalMatrix_, glmatrix);
}


static Rbt getEyePose()										{ return active_eye == SKY ? skyPose_ : cube_.pose_; }
static Rbt getArcballPose()									{ return active_object == SKY ? (active_camera_frame == WORLD_SKY ? Rbt() : skyPose_) : cube_.pose_; }


static void drawArcBall()
{
    sendModelViewNormalMatrix((getEyePose().getInverse() * getArcballPose()).getMatrix());
    glutWireSphere(arcball_radius, 20, 20);													// draw wireframe sphere
}

static void init_normals(Mesh& m)
{
    for (int i = 0; i < m.getNumVertices(); ++i) m.getVertex(i).setNormal(Vector3(0));
    for (int i = 0; i < m.getNumFaces(); ++i)
    {
        const Vector3 n = m.getFace(i).getNormal();
        if (std::abs(Vector3::dot(n, n) - 1) < 1e-6)										// we only use valid normals
        for (int j = 0; j < m.getFace(i).getNumVertices(); ++j) m.getFace(i).getVertex(j).setNormal(m.getFace(i).getVertex(j).getNormal() + n);
    }
    for (int i = 0; i < m.getNumVertices(); ++i) m.getVertex(i).setNormal(m.getVertex(i).getNormal().length2() < 1e-10 ? Vector3() : m.getVertex(i).getNormal().normalize());
}


static void draw_vertex(const Vector3& v, const Vector3& n)
{
    glNormal3f(n[0], n[1], n[2]);
    glVertex3f(v[0], v[1], v[2]);
}


static void draw(Mesh& m, const Vector3& LightE)
{
	GLfloat glmatrix[16];
	const Matrix4 projmat = Matrix4::makeProjection(frust_fovy, window_width / static_cast <double> (window_height), frust_near, frust_far);
	projmat.writeToColumnMajorMatrix(glmatrix);

	const Rbt eyeInverse = getEyePose().getInverse();
    sendModelViewNormalMatrix((eyeInverse * cube_.pose_).getMatrix());
    init_normals(m);

    glEnable(GL_CULL_FACE);
	glBegin(GL_TRIANGLES);
	 

    for (int i = 0; i < m.getNumFaces(); ++i)							// draw the base mesh
    {
        safe_glVertexAttrib3f(SState[0].h_colorAmbient_, 0.45, 0.3, 0.3);
        safe_glVertexAttrib3f(SState[0].h_colorDiffuse_, 0.2, 0.2, 0.2);
        draw_vertex(m.getFace(i).getVertex(0).getPosition(), m.getFace(i).getVertex(0).getNormal());
        safe_glVertexAttrib3f(SState[0].h_colorAmbient_, 0.45, 0.3, 0.3);
        safe_glVertexAttrib3f(SState[0].h_colorDiffuse_, 0.2, 0.2, 0.2);
        draw_vertex(m.getFace(i).getVertex(1).getPosition(), m.getFace(i).getVertex(1).getNormal());
        safe_glVertexAttrib3f(SState[0].h_colorAmbient_, 0.45, 0.3, 0.3);
        safe_glVertexAttrib3f(SState[0].h_colorDiffuse_, 0.2, 0.2, 0.2);
        draw_vertex(m.getFace(i).getVertex(2).getPosition(), m.getFace(i).getVertex(2).getNormal());
	}
	glEnd();

    glEnable(GL_BLEND);													// for the rest of the drawing (shells) enable alpha blending
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);

    if (draw_shells)
    {
        active_shader = 1;
        safe_glUseProgram(SState[active_shader].h_program_);
    	safe_glUniformMatrix4fv(SState[active_shader].h_uProjMatrix_, glmatrix);
    	safe_glUniform3f(SState[active_shader].h_uLight_, LightE[0], LightE[1], LightE[2]);
        sendModelViewNormalMatrix((eyeInverse * cube_.pose_).getMatrix());
    	safe_glUniform1i(SState[active_shader].h_texUnit0_, 0);
    	safe_glUniform1i(SState[active_shader].h_texUnit1_, 1);
    	glDepthMask(GL_TRUE);   // enables writing to depth-buffer
		Mesh shell_m(m);
		for (int i = 0; i < shell_m.getNumVertices(); i++) {
			shell_m.getVertex(i).setPosition(cube_.pose_ * tips[i]);
		}
		for (int i = 0; i < num_shells; ++i) {
			float h = i/(float)num_shells * fur_height;
			// TODO: before drawing each shell, set (valpha_exponent = 2 +
    		// 5*(height/fur_height)), where "height" is the height of the
    		// current shell, and fur_height is the total height of the
    		// fur.
			safe_glUniform1f(SState[active_shader].h_valpha_exponent_,2 + 5*(h/fur_height));
			// TODO: draw shells.
    		//
    		// VShader expects that you set (vtexCoord) (texture
    		// coordinates into the shell-texture) VShader also expects
    		// that you set gl_Normal (which will be the tangent direction
    		// of the hair)
			glBegin(GL_TRIANGLES);
			for (int j = 0; j < m.getNumFaces(); ++j)
			{
				Mesh::Face face = m.getFace(j);
				Mesh::Face shell_face = shell_m.getFace(j);
				for (int k = 0; k < 3; k++) {
				Mesh::Vertex v = face.getVertex(k);
				Vector3 p = v.getPosition();
				Vector3 s = p + v.getNormal()* fur_height;
				Vector3 t = shell_face.getVertex(k).getPosition();
				Vector3 n = (s - p)/num_shells;
				Vector3 d = (t - s) * (2. / (num_shells * (num_shells + 1)));
				Vector3 pi = p + (n * (i+1)) + d * ((i * (i-1)) / 2.);
				safe_glVertexAttrib2f(SState[active_shader].h_vtexCoord_,(k==1 || k==3)? 0 : texture_coord,(k==1 || k==2)? 0 : texture_coord);
				draw_vertex(pi, n + d * i);
				}
			}
			glEnd();
		}
    }
    glDepthMask(GL_TRUE);       // enables writing to depth-buffer
    glDisable(GL_BLEND);
    glEnable(GL_CULL_FACE);     // draw the shells
	
}


static void drawStuff()
{
    active_shader = 0;
    safe_glUseProgram(SState[active_shader].h_program_);

	GLfloat glmatrix[16];
	const Matrix4 projmat = Matrix4::makeProjection(frust_fovy, window_width / static_cast <double> (window_height), frust_near, frust_far);
	projmat.writeToColumnMajorMatrix(glmatrix);
	safe_glUniformMatrix4fv(SState[active_shader].h_uProjMatrix_, glmatrix);				// build & send proj. matrix to vshader
	
	const Rbt eyeInverse = getEyePose().getInverse();
	const Vector3 light = eyeInverse * Light;												// light direction in eye coordinates
	safe_glUniform3f(SState[active_shader].h_uLight_, light[0], light[1], light[2]);
	

	// draw floor 

	const Rbt groundFrame = Rbt();															// identity
	sendModelViewNormalMatrix((eyeInverse * groundFrame).getMatrix());
	

	// draw the floor
	glBegin(GL_TRIANGLES);
	safe_glVertexAttrib3f(SState[active_shader].h_colorAmbient_, 0.0, 0.0, 0.0);
	safe_glVertexAttrib3f(SState[active_shader].h_colorDiffuse_, 0.4, 1, 0.4);
	glNormal3f(0.0, 1.0, 0.0);
	glVertex3f(-floor_size, floor_y, -floor_size);

	safe_glVertexAttrib3f(SState[active_shader].h_colorAmbient_, 0.0, 0.0, 0.0);
	safe_glVertexAttrib3f(SState[active_shader].h_colorDiffuse_, 0.4, 1, 0.4);
	glNormal3f(0.0, 1.0, 0.0);
	glVertex3f( floor_size, floor_y,  floor_size);

	safe_glVertexAttrib3f(SState[active_shader].h_colorAmbient_, 0.0, 0.0, 0.0);
	safe_glVertexAttrib3f(SState[active_shader].h_colorDiffuse_, 0.4, 1, 0.4);
	glNormal3f(0.0, 1.0, 0.0);
	glVertex3f( floor_size, floor_y, -floor_size);
	
	safe_glVertexAttrib3f(SState[active_shader].h_colorAmbient_, 0.0, 0.0, 0.0);
	safe_glVertexAttrib3f(SState[active_shader].h_colorDiffuse_, 0.4, 1, 0.4);
	glNormal3f(0.0, 1.0, 0.0);
	glVertex3f(-floor_size, floor_y, -floor_size);

	safe_glVertexAttrib3f(SState[active_shader].h_colorAmbient_, 0.0, 0.0, 0.0);
	safe_glVertexAttrib3f(SState[active_shader].h_colorDiffuse_, 0.4, 1, 0.4);
	glNormal3f(0.0, 1.0, 0.0);
	glVertex3f(-floor_size, floor_y,  floor_size);

	safe_glVertexAttrib3f(SState[active_shader].h_colorAmbient_, 0.0, 0.0, 0.0);
	safe_glVertexAttrib3f(SState[active_shader].h_colorDiffuse_, 0.4, 1, 0.4);
	glNormal3f(0.0, 1.0, 0.0);
	glVertex3f( floor_size, floor_y,  floor_size);
	glEnd();

    if (display_arcball) drawArcBall();														// draw the arcball on top of everything else

    draw(mesh, light);
}



static void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear framebuffer color&depth

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


static Vector3 getArbBallDirection(Vector3 p, const double r)
{
    if (p[0]*p[0]+p[1]*p[1] > r*r) p *= std::sqrt((r*r - 1) / (p[0]*p[0]+p[1]*p[1]));		// in case the mouse moved outside the arcball
    return Vector3(p[0], p[1], std::sqrt(r*r - (p[0]*p[0]+p[1]*p[1]))).normalize();
}

static Rbt moveArcball(const Vector3& p0, const Vector3& p1)
{
    const Matrix4 projMatrix = Matrix4::makeProjection(frust_fovy, window_width / static_cast <double> (window_height), frust_near, frust_far);
	const Rbt eyeInverse = getEyePose().getInverse();
	const Vector3 arcballCenter = getArcballPose().getTranslation();
	
	Vector2 ballScreenCenter;
	double ballScreenRadius;
    if (!getScreenSpaceCircle(eyeInverse * arcballCenter, arcball_radius, projMatrix, frust_near, frust_fovy, window_width, window_height, &ballScreenCenter, &ballScreenRadius)) return Rbt();
    
    const Vector3 v0 = getArbBallDirection(p0 - Vector3(ballScreenCenter[0], ballScreenCenter[1], 0), ballScreenRadius);
    const Vector3 v1 = getArbBallDirection(p1 - Vector3(ballScreenCenter[0], ballScreenCenter[1], 0), ballScreenRadius);
    
    return Rbt(Quaternion(0.0, v1[0], v1[1], v1[2]) * Quaternion(0.0, v0[0], v0[1], v0[2]));
}


static Rbt do_Q_to_O_wrt_A(const Rbt& O, const Rbt& Q, const Rbt& A)
{
    return A * Q * A.getInverse() * O;
}

static void motion(const int x, const int y)
{
    if (!mouse_lclick_button && !mouse_rclick_button && !mouse_mclick_button) return;
    if (active_object == SKY && active_eye != SKY) return;									// we do not edit the eye when viewed from the objects

    const int nx = x;
    const int ny = window_height - y - 1;
    const double dx = nx - mouse_click_x;
    const double dy = ny - mouse_click_y;
    Rbt Q = mouse_lclick_button && !mouse_rclick_button ? Rbt(Quaternion::makeXRotation(-dy) * Quaternion::makeYRotation(dx)) : (
            mouse_rclick_button && !mouse_lclick_button ? Rbt(Vector3(dx, dy, 0) * 0.01) : 
                                                          Rbt(Vector3(0, 0, -dy) * 0.01));
    if ((active_object != SKY && active_object != active_eye) || (active_object == SKY && active_camera_frame == WORLD_SKY))
    if (mouse_lclick_button && !mouse_rclick_button)
    {
        Q = moveArcball(Vector3(mouse_click_x, mouse_click_y, 0), Vector3(nx, ny, 0));
    }
    if (active_object == active_eye)
    {
        if (active_eye != SKY || active_camera_frame != WORLD_SKY)
        {
            if (mouse_lclick_button && !mouse_rclick_button) Q = Q.getInverse();
        }
        else Q = Q.getInverse();
    }

    Rbt& O = active_object == SKY ? skyPose_ : cube_.pose_;
    const Rbt A(active_object == OBJECT0 ? cube_.pose_.getTranslation() : (			// the translation part of the reference frame
                active_camera_frame == WORLD_SKY ? Vector3() : skyPose_.getTranslation()),
                active_eye == OBJECT0 ? cube_.pose_.getRotation() :		 			// the rotation part of the reference frame
                                        skyPose_.getRotation());
    O = do_Q_to_O_wrt_A(O, Q, A);
    
    mouse_click_x = nx;
    mouse_click_y = ny;
    glutPostRedisplay();																	// we always redraw if we changed the scene
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
    display_arcball = state == GLUT_DOWN;
}


static void keyboard(const unsigned char key, const int x, const int y)
{
    switch (key)
    {
	    case 27: exit(0);																	// ESC
        case 'h':
            std::cerr << " ============== H E L P ==============\n\n";
            std::cerr << "h\t\thelp menu\n";
            std::cerr << "w\t\tsave screenshot\n";
            std::cerr << "s\t\tToggle draw-shells on/off.\n";
            std::cerr << "o\t\tCycle object to edit\n";
            std::cerr << "v\t\tCycle view\n";
            std::cerr << "drag left mouse to rotate\n";
            break;
        case 'w':
        	glFlush();
        	WritePPMScreenshot(window_width, window_height, "out.ppm");
            break;
	    case 's': draw_shells = !draw_shells; break;
        case 'v':
            active_eye = (active_eye+1) % 2;
            std::cerr << "Active eye is " << (active_eye == SKY ? "sky eye\n" : (active_eye == OBJECT0 ? "object0 eye\n" : "object1 eye\n"));
            break;
        case 'o':
            active_object = (active_object+1) % 2;
            std::cerr << "Active object is " << (active_object == SKY ? "sky eye\n" : (active_object == OBJECT0 ? "object0\n" : "object1\n"));
            break;
        case 'm':
            active_camera_frame = (active_camera_frame+1) % 2;
            std::cerr << "Editing sky eye w.r.t. " << (active_camera_frame == WORLD_SKY ? "world-sky frame\n" : "sky-sky frame\n");
            break;
		case '-':
		case '_':
			stiffness /= 1.05;
			cout<<"stiffness: "<<stiffness<<endl;
			break;
		case '=':
		case '+':
			stiffness *= 1.05;
			cout<<"stiffness: "<<stiffness<<endl;
			break;
		case 'z':
			if (damping > .93) {
				damping /= 1.001;
			}
			cout<<"damping: "<<damping<<endl;
			break;
		case 'x':
			if (damping < 0.999) {
				damping *= 1.001;
			} else {
				damping = 0.999;
			}
			cout<<"damping: "<<damping<<endl;
			break;
    }
    glutPostRedisplay();
}

static void special_keyboard(const int key, const int x, const int y)
{
    switch (key)
    {
	case GLUT_KEY_RIGHT: fur_height *= 1.05; cout<<"fur height: "<<fur_height<<endl; break;
        case GLUT_KEY_LEFT:  fur_height /= 1.05; cout<<"fur height: "<<fur_height<<endl; break;
        case GLUT_KEY_UP: texture_coord *= 1.05; cout<<"texture coord: "<<texture_coord<<endl; break;
        case GLUT_KEY_DOWN: texture_coord /= 1.05; cout<<"texture coord: "<<texture_coord<<endl; break;
    }
    glutPostRedisplay();
}


static void idle()
{
	 //  TODO:  animate the dynamics here
	for (int i = 0; i < num_sim_idle; i++) {
		for (int j = 0; j < mesh.getNumVertices(); j ++) {
			Vector3 p_world = cube_.pose_/*.getInverse()*/ * mesh.getVertex(j).getPosition();
			Vector3 s_world = cube_.pose_/*.getInverse()*/ * mesh.getVertex(j).getPosition() +
				 mesh.getVertex(j).getNormal()* fur_height;
			Vector3 a = gravity + ((s_world - tips[j]) * stiffness);

			tips[j] = tips[j] + vtips[j] * time_step;
			tips[j] = p_world + ((tips[j] - p_world)/(tips[j] - p_world).length()) * fur_height;
			vtips[j] = (vtips[j] + a * time_step) * damping;
		}
	}
    
    glutPostRedisplay();
}


static void initGlutState(int argc, char * argv[])
{
  glutInit(&argc, argv);																	// initialize Glut based on cmd-line args
  glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);									//  RGBA pixel channels and double buffering
  glutInitWindowSize(window_width, window_height);											// create a window
  glutCreateWindow("Assignment 9");															// title the window
  
  glutDisplayFunc(display);																	// display rendering callback
  glutReshapeFunc(reshape);																	// window reshape callback
  glutMotionFunc(motion);																	// mouse movement callback
  glutMouseFunc(mouse);																		// mouse click callback
  glutKeyboardFunc(keyboard);
  glutSpecialFunc(special_keyboard);
  glutIdleFunc(idle);
}

static void InitGLState()
{
  glClearColor(128./255., 200./255., 255./255., 0.);
  glClearDepth(0.);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glCullFace(GL_BACK);
//  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_GREATER);
  glReadBuffer(GL_BACK);

	// set up texture
	safe_glActiveTexture(GL_TEXTURE0);
	int twidth, theight, maxcol;
	FILE * f = std::fopen("textures/shell.ppm", "r");
	if (!f) { std::cerr << "Unable to open shell.ppm for reading. Exiting...\n"; exit(1); }
	std::fscanf(f, "P3 %d %d %d", &twidth, &theight, &maxcol);
	assert(maxcol == 255);
	char *pixdata = new char[twidth*theight*3];
	for (int i = 0; i < twidth*theight; ++i)
	{
	    int r, g, b;
	    std::fscanf(f, "%d %d %d", &r, &g, &b);
	    pixdata[3*i+0] = r;
	    pixdata[3*i+1] = g;
	    pixdata[3*i+2] = b;
    }
	glGenTextures(1, &h_texture_shells);
	glBindTexture(GL_TEXTURE_2D, h_texture_shells);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, twidth, theight, 0, GL_RGB, GL_UNSIGNED_BYTE, pixdata);
	delete [] pixdata;

	//second texture
	safe_glActiveTexture(GL_TEXTURE1);
	f = std::fopen("textures/fin.ppm", "r");
	if (!f) { std::cerr << "Unable to open fin.ppm for reading. Exiting...\n"; exit(1); }
	std::fscanf(f, "P3 %d %d %d", &twidth, &theight, &maxcol);
	assert(maxcol == 255);
	pixdata = new char[twidth*theight*3];
	for (int i = 0; i < twidth*theight; ++i)
	{
	    int r, g, b;
	    std::fscanf(f, "%d %d %d", &r, &g, &b);
	    pixdata[3*i+0] = r;
	    pixdata[3*i+1] = g;
	    pixdata[3*i+2] = b;
    }
	glGenTextures(1, &h_texture_fins);
	glBindTexture(GL_TEXTURE_2D, h_texture_fins);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, twidth, theight, 0, GL_RGB, GL_UNSIGNED_BYTE, pixdata);
	delete [] pixdata;
}


int main(int argc, char * argv[])
{
    mesh.load("bunny.mesh");//"ico.vp");													// for tests you can use "ico.vp" which is a 20-triangle mesh
    init_normals(mesh);
    init_tips();
	//for (int i = 0; i < mesh.getNumVertices(); i++){
	//	int valence = 0;
	//	Mesh::VertexIterator it = mesh.getVertex(i).getIterator(), it0(it);
	//	do {
	//		valence++;
	//		++it;
	//		Vector3 tmp = it.getVertex().getPosition();
	//		Vector3 tmp2 = it0.getVertex().getPosition();
	//	} while (it != it0);
	//}
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
        std::cerr << "Building shader: " << i << std::endl;
        const int shadeRet = ShaderInit(shader_file[i][0], shader_file[i][1], &SState[i].h_program_);
        if (!shadeRet)
        {
            std::cerr << "Error: could not build the shaders " << shader_file[i][0] << ", and " << shader_file[i][1] << ". Exiting...\n";
            assert(0);
        }        
        SState[i].h_uLight_ = safe_glGetUniformLocation(SState[i].h_program_, "uLight");	// Retrieve handles to variables in program
        SState[i].h_uProjMatrix_ = safe_glGetUniformLocation(SState[i].h_program_, "uProjMatrix");
        SState[i].h_uModelViewMatrix_ = safe_glGetUniformLocation(SState[i].h_program_, "uModelViewMatrix");
        SState[i].h_uNormalMatrix_ = safe_glGetUniformLocation(SState[i].h_program_, "uNormalMatrix");
        SState[i].h_texUnit0_ = safe_glGetUniformLocation(SState[i].h_program_, "texUnit0");
        SState[i].h_texUnit1_ = safe_glGetUniformLocation(SState[i].h_program_, "texUnit1");
        SState[i].h_vtexCoord_ = safe_glGetAttribLocation(SState[i].h_program_, "vtexCoord");
        SState[i].h_vTangent_ = safe_glGetAttribLocation(SState[i].h_program_, "vTangent");
        SState[i].h_valpha_exponent_ = safe_glGetUniformLocation(SState[i].h_program_, "valpha_exponent");
        SState[i].h_colorAmbient_ = safe_glGetAttribLocation(SState[i].h_program_, "colorAmbient");
        SState[i].h_colorDiffuse_ = safe_glGetAttribLocation(SState[i].h_program_, "colorDiffuse");        
    }
    
    glutMainLoop();																			// Pass control to glut to handle	the	main program loop
    return 0;
}


