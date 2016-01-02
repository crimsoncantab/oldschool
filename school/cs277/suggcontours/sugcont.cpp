////////////////////////////////////////////////////////////////////////
//
//   Loren McGinnis
//	 Harvard University
//   CS277 : Geometric Modeling in Computer Graphics
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


// ----------- GL stuff

static const double frust_fovy = 60.0;														// 60 degree field of view in y direction
static const double frust_near = -0.1;														// near plane
static const double frust_far = -50.0;														// far plane

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

static bool draw_shape = true;
static bool draw_silhouettes = true;
static bool draw_sug_cont = true;
static bool draw_grid = false;
static bool draw_princ_curvature = false;
static bool draw_all_k_0 = false;
static bool draw_zero_derivative = false;

static float derivative_threshold = 1.;
static const float theta_threshold = 0.05;

static double arcball_radius = 2.5;

static Mesh mesh;

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

static std::vector <ShaderState> SState(1);													// initializes a vector with 2 ShaderState

static const char * const shader_file[1][2] =
{
    {"./shaders/shader.vert", "./shaders/shader.frag"}
};
static const Vector3 Light(10.0, 5.5, 17.0);												// light position
struct Cube
{
    Rbt pose_;
    Vector3 color_;
    Cube()														{}
    Cube(const Rbt& p, const Vector3& c) : pose_(p), color_(c) 	{}
};

static Rbt skyPose_(Vector3(0.0, 0.5, 5.0));
static Cube cube_ = Cube(Rbt(Vector3(0.0, 0.0, 0.0)), Vector3(1, 0, 0));


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


static void init_curvature(Mesh& m) {
	for (int i = 0; i < m.getNumVertices(); i++) {
		Mesh::Vertex v = m.getVertex(i);
		Vector3 n = v.getNormal(), p = v.getPosition();
		Mesh::VertexIterator it = v.getIterator(), it0(it);
		//figure out valence for allocating correct array sizes
		int valence = 0;
		do {
			valence++;
			++it;
			Vector3 tmp = it.getVertex().getPosition();
			Vector3 tmp2 = it0.getVertex().getPosition();
		} while (it != it0);
		//arrays to remember important values at each neighbor
		float * norm_curv_i = new float[valence];
		Vector3 * t_i = new Vector3[valence];
		//the largest normal curvature will become a basis for temp. coords
		Vector3 e1_hat;
		float norm_curv_max = 0;
		int j = 0;
		do {
			Vector3 p_pi = it.getVertex().getPosition() - p;
			Vector3 n_ni = it.getVertex().getNormal() - n;
			norm_curv_i[j] = Vector3::dot(p_pi, n_ni) / Vector3::dot(p_pi,p_pi);
			t_i[j] = (p_pi - (n * Vector3::dot(p_pi, n))).normalize();
			//update max normal curvature
			if (i == 0 || norm_curv_i[j] > norm_curv_max) {
				norm_curv_max = norm_curv_i[j];
				e1_hat = t_i[j];
			}
			++it;
			j++;
		}while (it != it0);
		//get orthogonal unit vector to complete basis
		Vector3 e2_hat = Vector3::cross(e1_hat, n);
		//values to help find principle
		float a = norm_curv_max, a11 = 0, a12 = 0, a22 = 0, a13 = 0, a23 = 0;
		j = 0;
		do{
			float cos_theta = Vector3::dot(t_i[j], e1_hat), cos_2_theta = cos_theta * cos_theta;
			float sin_2_theta = 1 - cos_2_theta, sin_theta = sqrt(sin_2_theta);
			a11 += cos_2_theta * sin_2_theta;
			a12 += cos_theta * sin_2_theta * sin_theta;
			a22 += sin_2_theta * sin_2_theta;
			a13 += (norm_curv_i[j] - (a * cos_2_theta)) * cos_theta * sin_theta;
			a23 += (norm_curv_i[j] - (a * cos_2_theta)) * sin_2_theta;
			it++;
			j++;
		} while (it != it0);
		float b = ((a13 * a22) - (a23 * a12))/((a11 * a22) - (a12 * a12)), c = ((a11 * a23) - (a12 * a13))/((a11 * a22) - (a12 * a12));
		float gaussian = a * c - (b * b / 4), mean = (a + c) / 2;
		float k1 = mean + sqrt(mean * mean - gaussian);
		float k2 = mean - sqrt(mean * mean - gaussian);
		float theta_0 = 0.5 * asin(b / (k2 - k1));
		Vector3 e1 = e1_hat * cos(theta_0) + e2_hat * sin(theta_0);
		Vector3 e2 = e2_hat * cos(theta_0) - e1_hat * sin(theta_0);

		v.setE1(e1);
		v.setE2(e2);
		v.setK1(k1);
		v.setK2(k2);

		delete[] norm_curv_i; delete[] t_i;
	}
}


static void draw_vertex(const Vector3& v, const Vector3& n)
{
    glNormal3f(n[0], n[1], n[2]);
    glVertex3f(v[0], v[1], v[2]);
}

static void draw_edges(Mesh& m) {
	if (draw_grid || draw_silhouettes) {
		for (int i = 0; i < m.getNumEdges(); i++) {
			Mesh::Edge e = m.getEdge(i);
			Vector3 camera_loc = cube_.pose_ * skyPose_.getTranslation();
			Vector3 v = (camera_loc-((e.getVertex(0).getPosition() + e.getVertex(1).getPosition()) / 2));
			bool draw_as_normal_edge = true;
			if (draw_silhouettes) {
				Vector3 n1 = e.getFace(0).getNormal();
				Vector3 n2 = e.is_boundary() ? e.getFace(0).getNormal() * -1 : e.getFace(1).getNormal();
				float dot1 = Vector3::dot(v, n1);
				float dot2 = Vector3::dot(v, n2);
				if (((dot1 <= 0 && dot2 >= 0) || (dot1 >= 0 && dot2 <= 0))) {
					draw_as_normal_edge = false;
					safe_glVertexAttrib3f(SState[0].h_colorAmbient_, 0., 0., 0.);
					safe_glVertexAttrib3f(SState[0].h_colorDiffuse_, 0., 0., 0.);
					draw_vertex(e.getVertex(0).getPosition(), e.getVertex(0).getNormal());
					draw_vertex(e.getVertex(1).getPosition(), e.getVertex(1).getNormal());
				}
			}
			if (draw_as_normal_edge && draw_grid) {
				safe_glVertexAttrib3f(SState[0].h_colorAmbient_, 1., 0., 0.);
				safe_glVertexAttrib3f(SState[0].h_colorDiffuse_, 0., 0., 0.);
				draw_vertex(e.getVertex(0).getPosition(), e.getVertex(0).getNormal());
				draw_vertex(e.getVertex(1).getPosition(), e.getVertex(1).getNormal());
			}
		}
	}	
}

static void calc_dwkr(Mesh & m) {
	for (int i = 0; i < m.getNumFaces(); i ++) {
		Mesh::Face f = m.getFace(i);
		Vector3 a = f.getVertex(0).getPosition(), b = f.getVertex(1).getPosition(),c = f.getVertex(2).getPosition();
		Vector3 w = (f.getVertex(0).getW() + f.getVertex(1).getW() + f.getVertex(2).getW()).normalize();
		Vector3 wn = w - f.getNormal() * (Vector3::dot(w, f.getNormal()));
		Vector3 ac = c-a;
		Vector3 ad = ac * (Vector3::dot(ac, (b-a)) / Vector3::dot(ac,ac));
		Vector3 d = a + ad;
		float delta_k_ac = f.getVertex(2).getKr() - f.getVertex(0).getKr(), kd = f.getVertex(0).getKr() + (ad.length() / ac.length() * delta_k_ac);
		Vector3 b1 = ac.normalize(), b2 = (b - d).normalize();
		float b1_slope = delta_k_ac / ac.length(), b2_slope = (f.getVertex(1).getKr() - kd) / (b - d).length();
		float l1 = Vector3::dot(wn, b1);
		float l2 = Vector3::dot(wn, b2);
		f.getVertex(0).addDwKr(l1 * b1_slope + l2 * b2_slope);
		f.getVertex(1).addDwKr(l1 * b1_slope + l2 * b2_slope);
		f.getVertex(2).addDwKr(l1 * b1_slope + l2 * b2_slope);
	}


}

static void calc_radial_curvs(Mesh &m) {
		Vector3 c = (cube_.pose_ * skyPose_.getTranslation());
		for (int i = 0; i < m.getNumVertices(); i++){
			Mesh::Vertex vert = m.getVertex(i);
			Vector3 v = c - vert.getPosition(), n = vert.getNormal();
			Vector3 w = v - (n * Vector3::dot(v, n));
			float cos_phi = Vector3::dot(w.normalize(), vert.getE1()), cos_2_phi = cos_phi * cos_phi;
			vert.setW(w.normalize());
			vert.setView(v);
			vert.setKr((vert.getK1() * cos_2_phi) + (vert.getK2() * (1-cos_2_phi)));
			vert.setDwKr(0);
		}
		calc_dwkr(m);
}
static bool passes_theta_test(Mesh::Face f) {
	for (int i =0 ; i < 3; i ++) {
		Vector3 v = f.getVertex(i).getView();
		if (acos(Vector3::dot(f.getVertex(i).getNormal(), v) / v.length()) < theta_threshold) return false;
	}
	return true;
}
static bool passes_dir_test(Mesh::Face f) {
	return (f.getVertex(0).getDwKr() > derivative_threshold && f.getVertex(1).getDwKr() > derivative_threshold && f.getVertex(2).getDwKr() > derivative_threshold);
}

static void draw_sugg_contours(Mesh& m) {
	calc_radial_curvs(m);
	for (int i = 0; i < m.getNumFaces(); i++) {
		Mesh::Face f = m.getFace(i);
		
		if (draw_all_k_0 || (passes_dir_test(f) && passes_theta_test(f))) {
			safe_glVertexAttrib3f(SState[0].h_colorAmbient_, 0.7, .7, .7);
			Mesh::Vertex ver0 = f.getVertex(0),ver1 = f.getVertex(1),ver2= f.getVertex(2);
			float rad_curv0 = ver0.getKr(),rad_curv1 = ver1.getKr(),rad_curv2 = ver2.getKr();
			Mesh::Vertex * diff;Mesh::Vertex *  same1;Mesh::Vertex *  same2; float rad_curv_diff, rad_curv_same1, rad_curv_same2;
			bool draw = true;
			if ((rad_curv0 <= 0 && rad_curv1 >= 0 && rad_curv2 >= 0) || (rad_curv0 >= 0 && rad_curv1 <= 0 && rad_curv2 <= 0))
				{diff = &ver0; same1 = &ver2; same2 = &ver1; rad_curv_diff = rad_curv0;rad_curv_same1 = rad_curv2;rad_curv_same2 = rad_curv1;}
			else if ((rad_curv1 <= 0 && rad_curv2 >= 0 && rad_curv0 >= 0) || (rad_curv1 >= 0 && rad_curv2 <= 0 && rad_curv0 <= 0))
				{diff = &ver1; same1 = &ver2; same2 = &ver0; rad_curv_diff = rad_curv1;rad_curv_same1 = rad_curv2;rad_curv_same2 = rad_curv0;}
			else if ((rad_curv2 <= 0 && rad_curv0 >= 0 && rad_curv1 >= 0) || (rad_curv2 >= 0 && rad_curv1 <= 0 && rad_curv0 <= 0))
				{diff = &ver2; same1 = &ver1; same2 = &ver0; rad_curv_diff = rad_curv2;rad_curv_same1 = rad_curv1;rad_curv_same2 = rad_curv0;}
			else {draw = false;}

			//find first point in line
			if (draw) {
			Vector3 edge = diff->getPosition() - same1->getPosition();
			float delta_rad = abs(rad_curv_diff - rad_curv_same1);
			Vector3 point1 = diff->getPosition() - (edge * (abs(rad_curv_diff) / delta_rad));
			//second point
			edge = diff->getPosition() - same2->getPosition();
			delta_rad = abs(rad_curv_diff - rad_curv_same2);
			Vector3 point2 = diff->getPosition() - (edge * (abs(rad_curv_diff) / delta_rad));

			draw_vertex(point1, f.getNormal());
			draw_vertex(point2, f.getNormal());
			}
		}
		if (draw_zero_derivative) {
			safe_glVertexAttrib3f(SState[0].h_colorAmbient_, 1., 0., 1.);
			Mesh::Vertex ver0 = f.getVertex(0),ver1 = f.getVertex(1),ver2= f.getVertex(2);
			float rad_curv0 = ver0.getDwKr(),rad_curv1 = ver1.getDwKr(),rad_curv2 = ver2.getDwKr();
			Mesh::Vertex * diff;Mesh::Vertex *  same1;Mesh::Vertex *  same2; float rad_curv_diff, rad_curv_same1, rad_curv_same2;
			bool draw = true;
			if ((rad_curv0 <= 0 && rad_curv1 >= 0 && rad_curv2 >= 0) || (rad_curv0 >= 0 && rad_curv1 <= 0 && rad_curv2 <= 0))
				{diff = &ver0; same1 = &ver2; same2 = &ver1; rad_curv_diff = rad_curv0;rad_curv_same1 = rad_curv2;rad_curv_same2 = rad_curv1;}
			else if ((rad_curv1 <= 0 && rad_curv2 >= 0 && rad_curv0 >= 0) || (rad_curv1 >= 0 && rad_curv2 <= 0 && rad_curv0 <= 0))
				{diff = &ver1; same1 = &ver2; same2 = &ver0; rad_curv_diff = rad_curv1;rad_curv_same1 = rad_curv2;rad_curv_same2 = rad_curv0;}
			else if ((rad_curv2 <= 0 && rad_curv0 >= 0 && rad_curv1 >= 0) || (rad_curv2 >= 0 && rad_curv1 <= 0 && rad_curv0 <= 0))
				{diff = &ver2; same1 = &ver1; same2 = &ver0; rad_curv_diff = rad_curv2;rad_curv_same1 = rad_curv1;rad_curv_same2 = rad_curv0;}
			else {draw = false;}

			//find first point in line
			if (draw) {
			Vector3 edge = diff->getPosition() - same1->getPosition();
			float delta_rad = abs(rad_curv_diff - rad_curv_same1);
			Vector3 point1 = diff->getPosition() - (edge * (abs(rad_curv_diff) / delta_rad));
			//second point
			edge = diff->getPosition() - same2->getPosition();
			delta_rad = abs(rad_curv_diff - rad_curv_same2);
			Vector3 point2 = diff->getPosition() - (edge * (abs(rad_curv_diff) / delta_rad));

			draw_vertex(point1, f.getNormal());
			draw_vertex(point2, f.getNormal());
			}
		}
	}
}

static void draw(Mesh& m, const Vector3& LightE)
{
	glEnable(GL_POLYGON_OFFSET_FILL);
	GLfloat glmatrix[16];
	const Matrix4 projmat = Matrix4::makeProjection(frust_fovy, window_width / static_cast <double> (window_height), frust_near, frust_far);
	projmat.writeToColumnMajorMatrix(glmatrix);

	const Rbt eyeInverse = getEyePose().getInverse();
    sendModelViewNormalMatrix((eyeInverse * cube_.pose_).getMatrix());

	if (draw_shape) {
		glBegin(GL_TRIANGLES);
		//safe_glVertexAttrib3f(SState[0].h_colorAmbient_, .9, .9, .9);
		//safe_glVertexAttrib3f(SState[0].h_colorDiffuse_, 0.2, 0.2, 0.2);
		safe_glVertexAttrib3f(SState[0].h_colorAmbient_, 0.45, 0.3, 0.3);
		safe_glVertexAttrib3f(SState[0].h_colorDiffuse_, 0.2, 0.2, 0.2);
		for (int i = 0; i < m.getNumFaces(); ++i)							// draw the base mesh
		{

			draw_vertex(m.getFace(i).getVertex(0).getPosition(), m.getFace(i).getVertex(0).getNormal());
			draw_vertex(m.getFace(i).getVertex(1).getPosition(), m.getFace(i).getVertex(1).getNormal());
			draw_vertex(m.getFace(i).getVertex(2).getPosition(), m.getFace(i).getVertex(2).getNormal());
		}
		glEnd();
	}

	glDisable(GL_POLYGON_OFFSET_FILL);

	glBegin(GL_LINES);
	safe_glVertexAttrib3f(SState[0].h_colorAmbient_, 0., 0., 0.);
	safe_glVertexAttrib3f(SState[0].h_colorDiffuse_, 0., 0., 0.);
	draw_edges(m);
	if (draw_sug_cont) draw_sugg_contours(m);
	if (draw_princ_curvature) {
		for (int i = 0; i < m.getNumVertices(); i ++) {
		safe_glVertexAttrib3f(SState[0].h_colorAmbient_, 0., 0., 1.);
		safe_glVertexAttrib3f(SState[0].h_colorDiffuse_, 0., 0., 1.);
			draw_vertex(m.getVertex(i).getPosition(), m.getVertex(i).getNormal());
			draw_vertex(m.getVertex(i).getPosition() + m.getVertex(i).getE1() * .01 * m.getVertex(i).getK1(), m.getVertex(i).getNormal());
			draw_vertex(m.getVertex(i).getPosition(), m.getVertex(i).getNormal());
			draw_vertex(m.getVertex(i).getPosition() + m.getVertex(i).getE2() * .01 * m.getVertex(i).getK2(), m.getVertex(i).getNormal());
		}
	}
	glEnd();
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
            std::cerr << "o\t\tCycle object to edit\n";
            std::cerr << "v\t\tCycle view\n";
            std::cerr << "e\t\tToggle shape drawing.\n";
            std::cerr << "c\t\tToggle contours.\n";
            std::cerr << "s\t\tToggle suggestive-contours.\n";
            std::cerr << "g\t\tToggle grid.\n";
            std::cerr << "p\t\tToggle principal curvatures.\n";
            std::cerr << "d\t\tToggle derivative check.\n";
            std::cerr << "d\t\tDraw DwKr=0.\n";
            std::cerr << "drag left mouse to rotate\n";
            break;
        case 'w':
        	glFlush();
        	WritePPMScreenshot(window_width, window_height, "out.ppm");
            break;
        case 'v':
            active_eye = (active_eye+1) % 2;
            std::cerr << "Active eye is " << (active_eye == SKY ? "sky eye\n" : (active_eye == OBJECT0 ? "object0 eye\n" : "object1 eye\n"));
            break;
        case 'm':
            active_camera_frame = (active_camera_frame+1) % 2;
            std::cerr << "Editing sky eye w.r.t. " << (active_camera_frame == WORLD_SKY ? "world-sky frame\n" : "sky-sky frame\n");
            break;
		case 'e':
			draw_shape = !draw_shape;
			break;
		case 'c':
			draw_silhouettes = !draw_silhouettes;
			break;
		case 's':
			draw_sug_cont = !draw_sug_cont;
			break;
		case 'g':
			draw_grid = !draw_grid;
			break;
		case 'p':
			draw_princ_curvature = !draw_princ_curvature;
			break;
		case 'd':
			draw_all_k_0 = !draw_all_k_0;
			break;
		case 'z':
			draw_zero_derivative= !draw_zero_derivative;
			break;
    }
    glutPostRedisplay();
}

static void special_keyboard(const int key, const int x, const int y)
{
    switch (key)
    {
        case GLUT_KEY_RIGHT: break;
        case GLUT_KEY_LEFT: break;
        case GLUT_KEY_UP: break;
        case GLUT_KEY_DOWN: break;
    }
    glutPostRedisplay();
}


static void initGlutState(int argc, char * argv[])
{
  glutInit(&argc, argv);																	// initialize Glut based on cmd-line args
  glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);									//  RGBA pixel channels and double buffering
  glutInitWindowSize(window_width, window_height);											// create a window
  glutCreateWindow("Suggestive Contours");															// title the window
  
  glutDisplayFunc(display);																	// display rendering callback
  glutReshapeFunc(reshape);																	// window reshape callback
  glutMotionFunc(motion);																	// mouse movement callback
  glutMouseFunc(mouse);																		// mouse click callback
  glutKeyboardFunc(keyboard);
  glutSpecialFunc(special_keyboard);
}

static void InitGLState()
{
  glClearColor(1., 1., 1., 0.);
  //glClearColor(128./255., 200./255., 255./255., 0.);
  glClearDepth(0.);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glCullFace(GL_BACK);
  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_GREATER);
  glLineWidth(2.);
  glReadBuffer(GL_BACK);
  glPolygonOffset(-5., 0.);

}


int main(int argc, char * argv[])
{
	mesh.load("pear.mesh");
    init_normals(mesh);
	init_curvature(mesh);
    
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