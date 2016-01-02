////////////////////////////////////////////////////////////////////////
//
//	 Harvard University
//   CS175 : Computer Graphics
//   Loren McGinnis
//   Professor Steven Gortler
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
static int active_shader = 1;
static int active_object = 0;
static int active_eye = SKY;
static int active_camera_frame = WORLD_SKY;
static int map_resolution = 256;
static float refraction_index = .9;

static double arcball_radius = 2.5;
static bool use_bunny = false;

static Mesh mesh;

struct ShaderState
{
    GLuint	h_program_;
    GLuint	h_texture_;
    
    GLint	h_uLight_;								/// handle to uniform variable for light vector
    GLint	h_uCamera_;								/// handle to uniform variable for camera vector
    GLint	h_uProjMatrix_;							/// handle to unifrom var for projection matrix
    GLint	h_uObjectMatrix_;							/// handle to unifrom var for object matrix
    GLint	h_uModelViewMatrix_;					/// handle to uniform var for model view matrices for Base
    GLint	h_uNormalMatrix_;						/// handle to uniform var for transforming normals for Base
	
	GLint	h_refractIndex_;						/// handle to attribute variable for index of refraction
    GLint	h_colorAmbient_;						/// handle to attribute variable for color of vertex
    GLint	h_colorDiffuse_;						/// handle to attribute variable for color of vertex
    GLint	h_vtexCoord_;							/// handle to uniform variable for light vector
};

static std::vector <ShaderState> SState(4);													// initializes a vector with 2 ShaderState

static const char * const shader_file[4][2] =
{
    {"./shaders/shader.vert", "./shaders/shader.frag"},	
	{"./shaders/bunny.vert", "./shaders/bunny.frag"},
    {"./shaders/bunnyreflect.vert", "./shaders/bunnyreflect.frag"},
    {"./shaders/bunnyrefract.vert", "./shaders/bunnyrefract.frag"}
};
Matrix4 projmat = Matrix4::makeProjection(frust_fovy, window_width / static_cast <double> (window_height), frust_near, frust_far);




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

static void sendModelViewNormalMatrix(const Matrix4& MVM)									// takes MVM and sends it (and normal matrix) to the vertex shader
{
    GLfloat glmatrix[16];
	MVM.writeToColumnMajorMatrix(glmatrix);													// send MVM
	safe_glUniformMatrix4fv(SState[0].h_uModelViewMatrix_, glmatrix);
	
	MVM.getNormalMatrix().writeToColumnMajorMatrix(glmatrix);								// send normal matrix
	safe_glUniformMatrix4fv(SState[0].h_uNormalMatrix_, glmatrix);
}


static Rbt getEyePose()										{ return active_eye == SKY ? skyPose_ : cube_.pose_; }
static Rbt getArcballPose()									{ return active_object == SKY ? (active_camera_frame == WORLD_SKY ? Rbt() : skyPose_) : cube_.pose_; }

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
    safe_glUseProgram(SState[active_shader].h_program_);
	GLfloat glmatrix[16];
	projmat.writeToColumnMajorMatrix(glmatrix);
	safe_glUniformMatrix4fv(SState[active_shader].h_uProjMatrix_, glmatrix);	
	safe_glUniform1i(SState[active_shader].h_texture_, 0);
    safe_glUniform3f(SState[active_shader].h_uLight_, LightE[0], LightE[1], LightE[2]);
	Vector3 cameraLoc = skyPose_.getTranslation();
	safe_glUniform3f(SState[active_shader].h_uCamera_, cameraLoc[0],cameraLoc[1],cameraLoc[2]);
	const Rbt eyeInverse = getEyePose().getInverse();
	(eyeInverse * cube_.pose_).getMatrix().writeToColumnMajorMatrix(glmatrix);													// send MVM
	safe_glUniformMatrix4fv(SState[active_shader].h_uModelViewMatrix_, glmatrix);
	cube_.pose_.getMatrix().writeToColumnMajorMatrix(glmatrix);
	safe_glUniformMatrix4fv(SState[active_shader].h_uObjectMatrix_, glmatrix);

	safe_glVertexAttrib3f(SState[active_shader].h_colorAmbient_, 0.75, 0.75, 0.75);
    safe_glVertexAttrib3f(SState[active_shader].h_colorDiffuse_, 0.6, 0.6, 0.6);
	safe_glVertexAttrib1f(SState[active_shader].h_refractIndex_, refraction_index);
	if (use_bunny) {
		glBegin(GL_TRIANGLES);
		 
		for (int i = 0; i < m.getNumFaces(); ++i)							// draw the base mesh
		{
			draw_vertex(m.getFace(i).getVertex(0).getPosition(), m.getFace(i).getVertex(0).getNormal());
			draw_vertex(m.getFace(i).getVertex(1).getPosition(), m.getFace(i).getVertex(1).getNormal());
			draw_vertex(m.getFace(i).getVertex(2).getPosition(), m.getFace(i).getVertex(2).getNormal());
		}
		glEnd();
	} else {
		glutSolidSphere(1.0, 100, 100);
	}
	
}


static void drawEnvironment(Matrix4 projection_matrix, Rbt eyeInverse)
{
    safe_glUseProgram(SState[0].h_program_);
    	
	GLfloat glmatrix[16];
	projection_matrix.writeToColumnMajorMatrix(glmatrix);
	safe_glUniformMatrix4fv(SState[0].h_uProjMatrix_, glmatrix);				// build & send proj. matrix to vshader
	
	const Vector3 light = eyeInverse * Light;												// light direction in eye coordinates
	safe_glUniform3f(SState[0].h_uLight_, light[0], light[1], light[2]);
	

	// draw floor 

	const Rbt groundFrame = Rbt();															// identity
	sendModelViewNormalMatrix((eyeInverse * groundFrame).getMatrix());
	

	// draw the floor
	glBegin(GL_TRIANGLES);
	safe_glVertexAttrib3f(SState[0].h_colorAmbient_, 0.0, 0.0, 0.0);
	safe_glVertexAttrib3f(SState[0].h_colorDiffuse_, 0.4, 1, 0.4);
	glNormal3f(0.0, 1.0, 0.0);
	glVertex3f(-floor_size, floor_y, -floor_size);

	glNormal3f(0.0, 1.0, 0.0);
	glVertex3f( floor_size, floor_y,  floor_size);

	glNormal3f(0.0, 1.0, 0.0);
	glVertex3f( floor_size, floor_y, -floor_size);
	
	glNormal3f(0.0, 1.0, 0.0);
	glVertex3f(-floor_size, floor_y, -floor_size);

	glNormal3f(0.0, 1.0, 0.0);
	glVertex3f(-floor_size, floor_y,  floor_size);

	glNormal3f(0.0, 1.0, 0.0);
	glVertex3f( floor_size, floor_y,  floor_size);
	glEnd();


	sendModelViewNormalMatrix((eyeInverse * Rbt(Vector3(-2, 2, 0))).getMatrix());
	safe_glVertexAttrib3f(SState[0].h_colorAmbient_, 1.0, 0.0, 0.0);
	safe_glVertexAttrib3f(SState[0].h_colorDiffuse_, 1, 0.4, 0.4);
	glutSolidCube(1);
	sendModelViewNormalMatrix((eyeInverse * Rbt(Vector3(0, 2, -2))).getMatrix());
	safe_glVertexAttrib3f(SState[0].h_colorAmbient_, 0.0, 0.0, 1.0);
	safe_glVertexAttrib3f(SState[0].h_colorDiffuse_, 0.4, 0.4, 1);
	glutSolidCube(1);
	sendModelViewNormalMatrix((eyeInverse * Rbt(Vector3(2, 2, 0))).getMatrix());
	safe_glVertexAttrib3f(SState[0].h_colorAmbient_, 0.0, 1.0, 0.0);
	safe_glVertexAttrib3f(SState[0].h_colorDiffuse_, 0.4, 1, 0.4);
	glutSolidCube(1);
	sendModelViewNormalMatrix((eyeInverse * Rbt(Vector3(0, -2, -2))).getMatrix());
	safe_glVertexAttrib3f(SState[0].h_colorAmbient_, 1.0, 1.0, 0.0);
	safe_glVertexAttrib3f(SState[0].h_colorDiffuse_, 1, 1, 0.4);
	glutSolidCube(1);
}

static void updateEnvMap() {
	int twidth = map_resolution, theight = map_resolution;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, twidth, theight);
	Matrix4 snapMatrix = Matrix4::makeProjection(90.0,1, frust_near, frust_far);

	Rbt snapshotCam = cube_.pose_;
	safe_glActiveTexture(GL_TEXTURE0);

	char * pixdata = new char[twidth * theight * 3];

	drawEnvironment(snapMatrix, snapshotCam.setRotation(Quaternion::makeZRotation(180)).getInverse());
	glFlush();
	glReadPixels(0,0,twidth,theight,GL_RGB, GL_UNSIGNED_BYTE, pixdata);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGB, twidth, theight, 0, GL_RGB, GL_UNSIGNED_BYTE, pixdata);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	drawEnvironment(snapMatrix, snapshotCam.setRotation(Quaternion::makeXRotation(180)).getInverse());
	glFlush();
	glReadPixels(0,0,twidth,theight,GL_RGB, GL_UNSIGNED_BYTE, pixdata);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGB, twidth, theight, 0, GL_RGB, GL_UNSIGNED_BYTE, pixdata);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	drawEnvironment(snapMatrix, snapshotCam.setRotation(Quaternion::makeXRotation(-90)).getInverse());
	glFlush();
	glReadPixels(0,0,twidth,theight,GL_RGB, GL_UNSIGNED_BYTE, pixdata);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGB, twidth, theight, 0, GL_RGB, GL_UNSIGNED_BYTE, pixdata);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	drawEnvironment(snapMatrix, snapshotCam.setRotation(Quaternion::makeXRotation(90)).getInverse());
	glFlush();
	glReadPixels(0,0,twidth,theight,GL_RGB, GL_UNSIGNED_BYTE, pixdata);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGB, twidth, theight, 0, GL_RGB, GL_UNSIGNED_BYTE, pixdata);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	drawEnvironment(snapMatrix, snapshotCam.setRotation(Quaternion::makeYRotation(-90) * Quaternion::makeZRotation(180)).getInverse());
	glFlush();
	glReadPixels(0,0,twidth,theight,GL_RGB, GL_UNSIGNED_BYTE, pixdata);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGB, twidth, theight, 0, GL_RGB, GL_UNSIGNED_BYTE, pixdata);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	drawEnvironment(snapMatrix, snapshotCam.setRotation(Quaternion::makeYRotation(90) * Quaternion::makeZRotation(180)).getInverse());
	glFlush();
	glReadPixels(0,0,twidth,theight,GL_RGB, GL_UNSIGNED_BYTE, pixdata);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGB, twidth, theight, 0, GL_RGB, GL_UNSIGNED_BYTE, pixdata);

	glViewport(0,0,window_width, window_height);
	delete pixdata;
}


static void display()
{
	updateEnvMap();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear framebuffer color&depth

	Rbt eyeInverse = getEyePose().getInverse();
	drawEnvironment(projmat, eyeInverse);
	draw(mesh, eyeInverse * Light);

    glutSwapBuffers();																		// show the back buffer (where we rendered stuff)

    const GLenum errCode = glGetError();													// check for errors
    if (errCode != GL_NO_ERROR) std::cerr << "Error: " << gluErrorString(errCode) << std::endl;
}
static void reshape(const int w, const int h)
{
    window_width = w;
    window_height = h;
	projmat = Matrix4::makeProjection(frust_fovy, window_width / static_cast <double> (window_height), frust_near, frust_far);
    glViewport(0, 0, w, h);
    std::cerr << "Size of window is now " << w << "x" << h << std::endl;
    glutPostRedisplay();
}


static Vector3 getArcBallDirection(Vector3 p, const double r)
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
    
    const Vector3 v0 = getArcBallDirection(p0 - Vector3(ballScreenCenter[0], ballScreenCenter[1], 0), ballScreenRadius);
    const Vector3 v1 = getArcBallDirection(p1 - Vector3(ballScreenCenter[0], ballScreenCenter[1], 0), ballScreenRadius);
    
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
}


static void keyboard(const unsigned char key, const int x, const int y)
{
    switch (key)
    {
	    case 27: exit(0);																	// ESC
        case 'w':
        	glFlush();
        	WritePPMScreenshot(window_width, window_height, "out.ppm");
            break;
        case 'o':
            active_object = (active_object+1) % 2;
            std::cerr << "Active object is " << (active_object == SKY ? "sky eye\n" : (active_object == OBJECT0 ? "object0\n" : "object1\n"));
            break;
        case 'm':
            active_camera_frame = (active_camera_frame+1) % 2;
            std::cerr << "Editing sky eye w.r.t. " << (active_camera_frame == WORLD_SKY ? "world-sky frame\n" : "sky-sky frame\n");
            break;
		case ' ':
			use_bunny = !use_bunny;
			break;
		case 's':
			active_shader++;
			if (active_shader == 4) active_shader = 1;
    }
    glutPostRedisplay();
}

static void glewCheck() {
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
}

static void initGlutState(int argc, char * argv[])
{
  glutInit(&argc, argv);																	// initialize Glut based on cmd-line args
  glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);									//  RGBA pixel channels and double buffering
  glutInitWindowSize(window_width, window_height);											// create a window
  glutCreateWindow("Final Project");															// title the window
  
  glutDisplayFunc(display);																	// display rendering callback
  glutReshapeFunc(reshape);																	// window reshape callback
  glutMotionFunc(motion);																	// mouse movement callback
  glutMouseFunc(mouse);																		// mouse click callback
  glutKeyboardFunc(keyboard);
}


static void initTexture() {
	GLuint texture_handle;
	glGenTextures(1, &texture_handle);
	safe_glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texture_handle);
	//got this stuff from man. page 442
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_REPEAT);
}
static void initGLState()
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
  initTexture();
}



static void initShaders() {
	for (std::size_t i = 0; i < SState.size(); ++i)
    {
        std::cerr << "Building shader: " << i << std::endl;
        const int shadeRet = ShaderInit(shader_file[i][0], shader_file[i][1], &SState[i].h_program_);
        if (!shadeRet)
        {
            std::cerr << "Error: could not build the shaders " << shader_file[i][0] << ", and " << shader_file[i][1] << ". Exiting...\n";
            assert(0);
        }        
        SState[i].h_uLight_ = safe_glGetUniformLocation(SState[i].h_program_, "uLight");
		SState[i].h_uCamera_ = safe_glGetUniformLocation(SState[i].h_program_, "uCamera");
		SState[i].h_texture_ = safe_glGetUniformLocation(SState[i].h_program_, "environMap");
        SState[i].h_uProjMatrix_ = safe_glGetUniformLocation(SState[i].h_program_, "uProjMatrix");
        SState[i].h_uObjectMatrix_ = safe_glGetUniformLocation(SState[i].h_program_, "uObjectMatrix");
        SState[i].h_uModelViewMatrix_ = safe_glGetUniformLocation(SState[i].h_program_, "uModelViewMatrix");
        SState[i].h_uNormalMatrix_ = safe_glGetUniformLocation(SState[i].h_program_, "uNormalMatrix");
        SState[i].h_vtexCoord_ = safe_glGetAttribLocation(SState[i].h_program_, "vtexCoord");
        SState[i].h_refractIndex_ = safe_glGetAttribLocation(SState[i].h_program_, "refractIndex");
        SState[i].h_colorAmbient_ = safe_glGetAttribLocation(SState[i].h_program_, "colorAmbient");
        SState[i].h_colorDiffuse_ = safe_glGetAttribLocation(SState[i].h_program_, "colorDiffuse");        
    }
}

int main(int argc, char * argv[])
{
    mesh.load("bunny.mesh");
    init_normals(mesh);
    
    initGlutState(argc,argv);  

	glewCheck();

    initGLState();																			// this is our own ftion for setting some GL state
  
    initShaders();
    
    glutMainLoop();																			// Pass control to glut to handle	the	main program loop
    return 0;
}


