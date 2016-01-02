#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <GL/glut.h>
//#include <GL/glx.h>
#include <set>
#include <queue>
#include "mesh2.h"
#include "vec.h"
#include <iostream>
#include <vector>
#include <gsl_linalg.h>

extern "C" {
	#include <taucs.h>
}
using namespace std;

enum edit_method { ARAP, SINGLE, DOUBLE };
static int defx(800), defy(600);
static Vector3 frame[3] = {Vector3 (1,0,0), Vector3 (0,1,0), Vector3 (0,0,1)};
static Vector3 origin(defx/2, defy/2, 0);

static bool lines = true;																								// FLAGS
static bool smooth_shading = false;
static bool lclick = false;
static bool rclick = false;
static bool mclick = false;
static bool select_mode = false;
static bool base_mode = false;
static bool move_mode = false;
static bool deselect_mode = false;
static bool solve = false;
static edit_method method = DOUBLE;

static float mouse_x(0), mouse_y(0);
static float click_x(0), click_y(0);

static Mesh mesh;																										// da mesh
static std::set <int> selected_vertices;// indices of selected vertices
static std::set <int> handle_vertices;// indices of movable selected vertices
static std::vector <int> unselected;
static std::map <std::pair <int, int>, double> weights;
static int max_valence = 0;
static void * F = NULL;
static int * perm;
static int * invperm;
static taucs_ccs_matrix * Aod;
static int num_v;
static gsl_matrix * U = gsl_matrix_alloc(3,3);
static gsl_matrix * V = gsl_matrix_alloc(3,3);
static gsl_vector * diag = gsl_vector_alloc(3);
static gsl_vector * work = gsl_vector_alloc(3);
static Vector3 colors[3] = {Vector3(0.4,0.15,0.3), Vector3(0.3,0.4,0.15),Vector3(0.15,0.3,0.4)};
static Vector3 translation(0);


static double get_weight(int i, int j) {
	pair<int,int> key(i,j);
	return weights[key];
}

static double get_both_weights(int i, int j) {
	return get_weight(i,j) + get_weight(j,i);
}

static void put_weight(int i, int j, double weight) {
	pair<int,int> key(i,j);
	weights[key] = weight;
}

static void calc_weights() {
	int num_f = mesh.getNumFaces();
	for (int i = 0; i < num_f; i++) {
		Mesh::Face f = mesh.getFace(i);
		Mesh::Vertex v1 = f.getVertex(0), v2 = f.getVertex(1), v3 = f.getVertex(2);
		Vector3 u12 = (v2.getPosition() - v1.getPosition()).normalize();
		Vector3 u23 = (v3.getPosition() - v2.getPosition()).normalize();
		Vector3 u31 = (v1.getPosition() - v3.getPosition()).normalize();
		put_weight(v1.v_, v2.v_, Vector3::dot(u23 * -1, u31) / (Vector3::cross(u23 * -1, u31).length()));
		put_weight(v2.v_, v3.v_, Vector3::dot(u31 * -1, u12) / (Vector3::cross(u31 * -1, u12).length()));
		put_weight(v3.v_, v1.v_, Vector3::dot(u12 * -1, u23) / (Vector3::cross(u12 * -1, u23).length()));
	}
}

static void reset() {
	for (int i = 0; i < mesh.getNumVertices(); i++) {
		Mesh::Vertex v = mesh.getVertex(i);
		v.setPosition(v.getOldPosition());
	}
	selected_vertices.clear();
	handle_vertices.clear();
	solve = false;
	frame[0] = Vector3 (1,0,0); frame[1] = Vector3 (0,1,0); frame[2] = Vector3 (0,0,1);
	origin = Vector3 (defx/2, defy/2, 0);
}
static Vector3 rotate(const Vector3& p)
{
    return Vector3 (p[0]*frame[0][0] + p[1]*frame[1][0] + p[2]*frame[2][0],
                             p[0]*frame[0][1] + p[1]*frame[1][1] + p[2]*frame[2][1],
                             p[0]*frame[0][2] + p[1]*frame[1][2] + p[2]*frame[2][2]);
}

static Vector3 transform(const Vector3& p)
{
    return rotate(p) + origin;
}

template <class T> static T sqr(const T& x)				{ return x*x; }

static Vector3 color(const Vector3& n)
{
    return Vector3 (0.2f) * sqr(sqr(n[2])) + colors[method] * sqr(n[2]) + Vector3 (0.25f);
}

static void color(const Vector3& tn, const Vector3& vn)
{
    const Vector3 c = !smooth_shading ? color(tn) : color(vn);
    glColor3f(c[0], c[1], c[2]);
}


int my_find(const vector<int> list, int value) {
	int size = list.size();
	for (int i = 0; i < size; i ++){
		if (list[i] == value) return i;
	}
	return -1;
}
void prefactor() {
	if (F != NULL) {
		taucs_supernodal_factor_free(F);
		taucs_ccs_free(Aod);
		unselected.clear();
		if (selected_vertices.empty()) {
			F = NULL;
			return;
		}
	}
	int total_v = mesh.getNumVertices();
	num_v = total_v- (selected_vertices.size());
	unselected.resize(num_v);
	int un_index = 0;
	for (int i = 0; i < total_v; i ++){
		if (selected_vertices.find(i) == selected_vertices.end()) {
			unselected[un_index] = i;
			un_index++;
		}
	}

	vector<int> ci(num_v+1);//location of column splits
	vector<int> ri(num_v * (max_valence + 1));//row indices
	vector<double> data(num_v * (max_valence + 1));
	int cur_data_i = 0;

	for (int i = 0; i < num_v; i ++) {
		ci[i] = cur_data_i;
		
		Mesh::Vertex v = mesh.getVertex(unselected[i]);
		int valence = v.getValence();
		int v_v = v.v_;
		priority_queue <int> neighbors;
		Mesh::VertexIterator it = v.getIterator(), it0(it);

		int diag_i = cur_data_i;
		cur_data_i++;
		data[diag_i] = 0;
		ri[diag_i] = i;
		
		//put unseleced vertices in pqueue to add to matrix
		do {
			Mesh::Vertex vert_j = it.getVertex();
			if (vert_j.v_ > v_v && selected_vertices.find(vert_j.v_) == selected_vertices.end()) {
				neighbors.push(vert_j.v_ * -1);
			} else {
				data[diag_i] += get_both_weights(v_v, vert_j.v_);
			}
			++it;
		} while (it != it0);

		while (!neighbors.empty()) {

			int vert_j = neighbors.top() * -1;
			neighbors.pop();
			
			float weight = get_both_weights(v_v, vert_j);
			data[diag_i] += weight;
			data[cur_data_i] = weight * -1.;
			int find = my_find(unselected, vert_j);
			assert(find != -1);
			ri[cur_data_i] = find;
			cur_data_i++;
		}
	}
	ci[num_v] = cur_data_i;
	ri.resize(cur_data_i);
	data.resize(cur_data_i);

	// create TAUCS matrix from vector objects an, jn and ia
	taucs_ccs_matrix * A = taucs_ccs_create(num_v,num_v,max_valence+1 /** num_v*/, TAUCS_DOUBLE | TAUCS_SYMMETRIC | TAUCS_LOWER);
	A->colptr = &ci[0];
	A->rowind = &ri[0];
	A->values.d = &data[0];
	

	//Using TAUCS low-level routines
	// 1) Reordering
	taucs_ccs_order(A, &perm, &invperm, "metis");
	Aod = taucs_ccs_permute_symmetrically(A, perm, invperm);

	// 2) Factoring
	F = taucs_ccs_factor_llt_mf(Aod);
	//taucs_ccs_free(A); //maybe we should do this later....
}
void solve_with_prefactor(taucs_double * b, taucs_double * x,int dim) {
	// we need solution and RHS twice: original and reordered
	// create TAUCS right-hand side
	vector <double> bodv(dim);
	taucs_double* bod = &*bodv.begin(); 

	// allocate TAUCS solution vector
	vector <double> xodv(dim);
	taucs_double* xod = &*xodv.begin(); 

	taucs_vec_permute(dim, TAUCS_DOUBLE, b, bod, perm);

	// 3) Back substitution and reodering the solution back
	taucs_supernodal_solve_llt(F, xod, bod);	
	taucs_vec_ipermute(dim, TAUCS_DOUBLE, xod, x, perm);
}
void lin_solve_mesh(int dir) {
	vector<double> b(num_v); //rhs
	vector<double> x(num_v);

	for (int i = 0; i < num_v; i ++) {
		
		Mesh::Vertex v = mesh.getVertex(unselected[i]);
		int valence = v.getValence();
		Mesh::VertexIterator it = v.getIterator(), it0(it), itold(it);
		double rhs = 0;

		if(method == SINGLE) {
			++it;
			++it0;
		}
		
		//build rhs, put other unseleced vertices in pqueue to add to matrix
		gsl_matrix * r_i;
		if (method == ARAP || method == DOUBLE) r_i = (gsl_matrix *) v.getRotation();
		do {
			Mesh::Vertex vert_j = it.getVertex();
			if (selected_vertices.find(vert_j.v_) != selected_vertices.end()) {
				rhs += vert_j.getPosition()[dir] * get_both_weights(v.v_, vert_j.v_);
			}
			Vector3 diff = v.getOldPosition() - vert_j.getOldPosition();
			if (method == ARAP) {
				gsl_matrix * r_j = (gsl_matrix *) vert_j.getRotation();
				Vector3 rotdiff =
					(Vector3(gsl_matrix_get(r_i, dir, 0),gsl_matrix_get(r_i, dir, 1),gsl_matrix_get(r_i, dir, 2))+
					Vector3(gsl_matrix_get(r_j, dir, 0),gsl_matrix_get(r_j, dir, 1),gsl_matrix_get(r_j, dir, 2))) * .5;
				rhs += Vector3::dot(rotdiff,(diff)) * get_both_weights(v.v_, vert_j.v_);
			} else if (method == SINGLE) {
				Mesh::Face new_face = it.getFace();
				Mesh::Face old_face = itold.getFace();
				gsl_matrix * r_ij = (gsl_matrix *) old_face.getRotation();
				gsl_matrix * r_ji = (gsl_matrix *) new_face.getRotation();
				Vector3 r_ij_dir = Vector3(gsl_matrix_get(r_ij, dir, 0),gsl_matrix_get(r_ij, dir, 1),gsl_matrix_get(r_ij, dir, 2));
				Vector3 r_ji_dir = Vector3(gsl_matrix_get(r_ji, dir, 0),gsl_matrix_get(r_ji, dir, 1),gsl_matrix_get(r_ji, dir, 2));
				rhs += (Vector3::dot(r_ij_dir, diff) * get_weight(vert_j.v_, v.v_) + Vector3::dot(r_ji_dir, diff) * get_weight(v.v_, vert_j.v_) );
				++itold;
			} else {
				gsl_matrix * r_j = (gsl_matrix *) vert_j.getRotation();
				Vector3 roti(gsl_matrix_get(r_i, dir, 0),gsl_matrix_get(r_i, dir, 1),gsl_matrix_get(r_i, dir, 2));
				Vector3 rotj(gsl_matrix_get(r_j, dir, 0),gsl_matrix_get(r_j, dir, 1),gsl_matrix_get(r_j, dir, 2));
				rhs += .5 * get_both_weights(v.v_, vert_j.v_) * (Vector3::dot(roti, diff) + Vector3::dot(rotj, diff));
			}
			++it;
		} while (it != it0);

		b[i] = rhs;
	}

	solve_with_prefactor(&b[0], &x[0], num_v);

	for (int i = 0; i < num_v; i++) {
		Vector3 pos = mesh.getVertex(unselected[i]).getPosition();
		pos[dir] = x[i];
		mesh.getVertex(unselected[i]).setPosition(pos);
	}
}

void print_matrix(gsl_matrix * m) {
	for (int k = 0; k < 3; k++){
		for (int j = 0; j < 3; j++){
			cout << gsl_matrix_get(m, k, j) << ",";
		}
		cout <<endl;
	}
}


double det_3_x_3(gsl_matrix * m) {
	double det = gsl_matrix_get(m,0,0) * gsl_matrix_get(m,1,1) *
	gsl_matrix_get(m,2,2) + gsl_matrix_get(m,0,1)
	* gsl_matrix_get(m,1,2) * gsl_matrix_get(m,2,0)
		+ gsl_matrix_get(m,0,2) * gsl_matrix_get(m,1,0) * gsl_matrix_get(m,2,1) - gsl_matrix_get(m,0,0) * gsl_matrix_get(m,1,2) * gsl_matrix_get(m,2,1)
		- gsl_matrix_get(m,0,1) * gsl_matrix_get(m,1,0) * gsl_matrix_get(m,2,2) - gsl_matrix_get(m,0,2) * gsl_matrix_get(m,1,1) * gsl_matrix_get(m,2,0);
	assert (det == 1 || det == -1);
	return det;
}

void mult_v_ut_into_rot(gsl_matrix * v, gsl_matrix * u, gsl_matrix * rot) {
	gsl_matrix * ut = gsl_matrix_alloc(3,3);
	int count = 0;
	int det = 0;
	do {
		if (count > 0) {
			gsl_matrix_set(u, 0, 2, -gsl_matrix_get(u,0,2));
			gsl_matrix_set(u, 1, 2, -gsl_matrix_get(u,1,2));
			gsl_matrix_set(u, 2, 2, -gsl_matrix_get(u,2,2));
		}

		gsl_matrix_transpose_memcpy(ut, u);

		for (int i =0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				double val = 0.;
				for (int k = 0; k < 3; k++) {
					val += gsl_matrix_get(v, i, k) * gsl_matrix_get(ut, k, j);
				}
				gsl_matrix_set(rot, i, j, val);
			}
		}

		count++;
		det = det_3_x_3(rot);
	} while (det != 1 && count < 1 );
	assert(det==1);
}


void update_rots() {

	if (method == ARAP) {
		int num_ver = mesh.getNumVertices();
		for (int i = 0; i < num_ver; i++) {
			gsl_matrix_set_zero(U);
			Mesh::Vertex v = mesh.getVertex(i);
			Mesh::VertexIterator it = v.getIterator(), it0(it);

			do {
				Mesh::Vertex vj = it.getVertex();
				Vector3 eijprime = v.getPosition() - vj.getPosition();
				Vector3 eij = v.getOldPosition() - vj.getOldPosition();
				//double w = get_both_weights(v.v_, vj.v_);
				double w = 1;
				for (int j = 0; j < 3; j++) {
					gsl_matrix_set(U,j,0,gsl_matrix_get(U,j,0)+eijprime[0] * eij[j] * w);
					gsl_matrix_set(U,j,1,gsl_matrix_get(U,j,1)+eijprime[1] * eij[j] * w);
					gsl_matrix_set(U,j,2,gsl_matrix_get(U,j,2)+eijprime[2] * eij[j] * w);
				}
				++it;
			} while (it != it0);
			//and now for the svd
			gsl_linalg_SV_decomp(U, V, diag, work);
			mult_v_ut_into_rot(V, U, (gsl_matrix *) v.getRotation());

		}
	} else if (method == SINGLE) {
		int num_face = mesh.getNumFaces();
		for (int i = 0; i < num_face; i++) {
			gsl_matrix_set_zero(U);
			Mesh::Face f = mesh.getFace(i);
			/*Vector3 vec(0), vecprime(0);
			for (int k = 0; k < 3; k++) {
				vec += f.getVertex(k).getOldPosition();
				vecprime += f.getVertex(k).getPosition();
			}
			vec /= 3;
			vecprime /= 3;*/

			for (int k = 0; k < 3; k++) {
				
				Mesh::Vertex v = f.getVertex(k), vj = f.getVertex((k+1)%3);
				Vector3 eijprime = v.getPosition() - vj.getPosition();
				Vector3 eij = v.getOldPosition() - vj.getOldPosition();
				//double w = get_both_weights(f.getVertex(k).v_, f.getVertex((k+1)%3).v_);
				double w = 1;
				for (int j = 0; j < 3; j++) {
					gsl_matrix_set(U,j,0,gsl_matrix_get(U,j,0)+eijprime[0] * eij[j] * w);
					gsl_matrix_set(U,j,1,gsl_matrix_get(U,j,1)+eijprime[1] * eij[j] * w);
					gsl_matrix_set(U,j,2,gsl_matrix_get(U,j,2)+eijprime[2] * eij[j] * w);
				}
			}
			
			//and now for the svd
			gsl_linalg_SV_decomp(U, V, diag, work);
			mult_v_ut_into_rot(V, U, (gsl_matrix *) f.getRotation());

		}
	} else {
		Vector3 center = mesh.get_old_center();
		Vector3 new_center = mesh.get_center();
		translation = new_center - center;

		int num_ver = mesh.getNumVertices();
		for (int i = 0; i < num_ver; i++) {
			gsl_matrix * rot = (gsl_matrix *) mesh.getVertex(i).getRotation();
			Vector3 oldv = (mesh.getVertex(i).getOldPosition() - center).normalize();
			Vector3 newv = (mesh.getVertex(i).getPosition() - new_center).normalize();

			Vector3 cross = Vector3::cross(oldv, newv);
			double s = cross.length();
			Vector3 a(0);
			if (s != 0) {
				a = cross.normalize();
			}
			double x = a[0], y = a[1], z = a[2];
			double c= Vector3::dot(oldv, newv);
			double C = 1 - c;

			gsl_matrix_set(rot,0,0,x*x*C+c);
			gsl_matrix_set(rot,0,1,x*y*C-z*s);
			gsl_matrix_set(rot,0,2,x*z*C+y*s);
			gsl_matrix_set(rot,1,0,y*x*C+z*s);
			gsl_matrix_set(rot,1,1,y*y*C+c);
			gsl_matrix_set(rot,1,2,y*z*C-x*s);
			gsl_matrix_set(rot,2,0,z*x*C-y*s);
			gsl_matrix_set(rot,2,1,z*y*C+x*s);
			gsl_matrix_set(rot,2,2,z*z*C+c);
		}
	}
}
void solve_mesh() {

	update_rots();

	lin_solve_mesh(0);
	lin_solve_mesh(1);
	lin_solve_mesh(2);
}
static void display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

				// viewing transformation
	glLoadIdentity();
//	gluLookAt( 0,0,-2,	// position
//			   0,0,0,	// aim
//			   0,1,0 );	// up direction
	
			// modeling transformation
//		m.rotate(2, 0, 0.1);

    glBegin(GL_TRIANGLES);
    for (int i = 0; i < mesh.getNumFaces(); ++i)
    {
		Mesh::Face f = mesh.getFace(i);
		const Vector3 v0 = transform(f.getVertex(0).getPosition());
        const Vector3 v1 = transform(f.getVertex(1).getPosition());
        const Vector3 v2 = transform(f.getVertex(2).getPosition());
        const Vector3 n0 = rotate(f.getVertex(0).getNormal()).normalize();
        const Vector3 n1 = rotate(f.getVertex(1).getNormal()).normalize();
        const Vector3 n2 = rotate(f.getVertex(2).getNormal()).normalize();
		const Vector3 n = rotate(f.getNormal());

        color(n, n0);
        glVertex3f(v0[0], v0[1], v0[2]);
        color(n, n1);
        glVertex3f(v1[0], v1[1], v1[2]);
        color(n, n2);
        glVertex3f(v2[0], v2[1], v2[2]);
    }
    glEnd();
    glBegin(GL_QUADS);
    glColor3f(1,0,0);
    for (std::set <int>::const_iterator it = selected_vertices.begin(); it != selected_vertices.end(); ++it)
    {
		if (handle_vertices.find(*it) == handle_vertices.end()) {
			glColor3f(0,0,1);
		} else {
			glColor3f(0,1,0);
		}
		const Vector3 v = transform(mesh.getVertex(*it).getPosition());
        static const int rad = 3;
        glVertex3f(v[0]-rad, v[1]-rad, v[2]);
        glVertex3f(v[0]+rad, v[1]-rad, v[2]);
        glVertex3f(v[0]+rad, v[1]+rad, v[2]);
        glVertex3f(v[0]-rad, v[1]+rad, v[2]);
    }
    glEnd();
    if (lines)
  	{
        glBegin(GL_LINES);
		glColor3f(0,0,0);
        for (int i = 0; i < mesh.getNumFaces(); ++i)
        {
            Mesh::Face f = mesh.getFace(i);
			const Vector3 v0 = transform(f.getVertex(0).getPosition());
            const Vector3 v1 = transform(f.getVertex(1).getPosition());
            const Vector3 v2 = transform(f.getVertex(2).getPosition());
            glVertex3f(v0[0], v0[1], v0[2]+1);
            glVertex3f(v1[0], v1[1], v1[2]+1);
            glVertex3f(v1[0], v1[1], v1[2]+1);
            glVertex3f(v2[0], v2[1], v2[2]+1);
            glVertex3f(v2[0], v2[1], v2[2]+1);
            glVertex3f(v0[0], v0[1], v0[2]+1);
        }
        glEnd();
    }
    if (select_mode && lclick && !rclick)
    {
    	glEnable(GL_BLEND);
    	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		if (deselect_mode) {
			glColor4f(0.5,0,0,0.5);
		} else if (base_mode) {
			glColor4f(0,0,0.5,0.5);
		} else {
			glColor4f(0,0.5,0,0.5);
		}
        glDisable(GL_DEPTH_TEST);
        glBegin(GL_QUADS);
        glVertex3f(click_x, click_y, 0);
        glVertex3f(mouse_x, click_y, 0);
        glVertex3f(mouse_x, mouse_y, 0);
        glVertex3f(click_x, mouse_y, 0);
        glEnd();
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
    }    
	glFlush();
	glutSwapBuffers();
}



void resize(int w, int h)
{
	origin = Vector3 (origin[0]*w/(float)defx, origin[1]*h/(float)defy, 0);
	defx = w;
	defy = h;
	glViewport(0,0,w,h);
	
		// we touch the projection matrix
  	glMatrixMode(GL_PROJECTION);	// GL_PROJECTION|MODELVIEW|TEXTURE
	glLoadIdentity();
//	gluPerspective(35.0, (float)w/(float)h, 1, 10000.0);

    glOrtho(0, defx, 0, defy, -1000, 1000); // x,y,z bounds
	glMatrixMode(GL_MODELVIEW);
	glutPostRedisplay();
}


static void mouse(int button, int state, int x, int y) 
{
    if (button == GLUT_LEFT_BUTTON) lclick = state == GLUT_DOWN;
    if (button == GLUT_RIGHT_BUTTON) rclick = state == GLUT_DOWN;
    if (button == GLUT_MIDDLE_BUTTON) mclick = state == GLUT_DOWN;
    if (state != GLUT_DOWN)
    {
        if (select_mode)
        {
            const int mx = min((float)x, click_x);
            const int Mx = max((float)x, click_x);
            const int my = min((float)defy-y-1, click_y);
            const int My = max((float)defy-y-1, click_y);
            for (int i = 0; i < mesh.getNumVertices(); ++i)
            {
				const Vector3 v = transform(mesh.getVertex(i).getPosition());
                if (mx <= v[0] && v[0] <= Mx && my <= v[1] && v[1] <= My)
                {
					if (deselect_mode) {
						selected_vertices.erase(i);
						handle_vertices.erase(i);
					} else {
						selected_vertices.insert(i);
						if (!base_mode) {
							handle_vertices.insert(i);
						} else {
							handle_vertices.erase(i);
						}
					}
                    
                }
            }
			prefactor();
        }
        if (select_mode || move_mode) std::cerr << "done.\n";
        mesh.init_normals();																					// must recompute the normals since the geometry changed
    }
    click_x = mouse_x = x;
    click_y = mouse_y = defy-y-1;
    glutPostRedisplay();
}


static void mousemove(int x, int y)
{

    const float dx = x - mouse_x;
    const float dy = defy-y-1 - mouse_y;
    
    if (!select_mode && !move_mode)
    {
        if (lclick && !rclick)														// rotate
        for (int i = 0; i < 3; ++i)
    	{
	        frame[i].rotate <2, 0> (dx * 0.015);
	        frame[i].rotate <2, 1> (dy * 0.015);
        }
        if (rclick && !lclick)														// translate (x,y)
        {
            origin += Vector3 (dx, dy, 0);
        }
        if (rclick && lclick)														// translate (z)
        {
            for (int i = 0; i < 3; ++i) frame[i] *= dx > 0 ? dx*0.02 : 1/(dx*0.02);
        }
    }
    mouse_x = x;
    mouse_y = defy-y-1;
    if (move_mode && lclick && !rclick)														// move
    {
        const Vector3 d__ = Vector3 (dx,dy,0) / frame[0].length();
        const Vector3 d(Vector3::dot(d__, frame[0]), Vector3::dot(d__, frame[1]), Vector3::dot(d__, frame[2]));
		for (std::set <int>::const_iterator it = handle_vertices.begin(); it != handle_vertices.end(); ++it) mesh.getVertex(*it).setPosition(mesh.getVertex(*it).getPosition() += d);
    }
    glutPostRedisplay();
}




static void print_help() {
	std::cerr << "h: show help\n";
	std::cerr << "left button: rotate\n";
    std::cerr << "right button: translate (x,y)\n";
    std::cerr << "middle button: translate (z)\n";
    std::cerr << "q/ESC: quit\n";
    std::cerr << "f: toggle faceted/smooth shading\n";
    std::cerr << "+/-: zoom in/out (for systems without third button)\n";
    std::cerr << "l: toggle wireframe\n";
    std::cerr << "s: select handle vertices mode (shown in GREEN)\n";
    std::cerr << "b: select base vertices mode (shown in BLUE)\n";
    std::cerr << "d: deselect mode\n";
    std::cerr << "z: toggle method (ARAP, Poisson, Bipoisson)\n";
    std::cerr << "g: real time editing\n";
    std::cerr << "t: single iteration editing\n";
    std::cerr << "r: reset mesh\n";
    std::cerr << "m: move selected vertices mode\n";
}
void keyboard(unsigned char key, int x, int y)
{

	if (key == 'q' || key == 27) exit(0);
	if (key == 'h') print_help();
	if (key == 'f') smooth_shading = !smooth_shading;
	if (key == '=' || key == '+') for (int i = 0; i < 3; ++i) frame[i] *= 1.05;
	if (key == '-' || key == '_') for (int i = 0; i < 3; ++i) frame[i] /= 1.05;
	if (key == 'l') lines = !lines;
	if (key == 's') { std::cerr << "Toggling select mode... " <<endl; select_mode = !select_mode; base_mode = move_mode = false; }
	if (key == 'b') { std::cerr << "Toggling select mode (base vertices)... " <<endl; base_mode = !base_mode; select_mode = base_mode; move_mode = false;}
	if (key == 'm') { std::cerr << "Toggling move mode... " <<endl; move_mode = !move_mode; select_mode = base_mode = false; }
	if (key == 'd') { deselect_mode = !deselect_mode; }
	if (key == 'z') { std::cerr << "Toggling method..." <<endl; method =(edit_method) (((int)method + 1) % 3); }
	if (key == 'g' && F != NULL) solve=!solve;
	if (key == 't' && F != NULL) solve_mesh();
	if (key == 'r') { reset(); }

	glutPostRedisplay();
}


void set_max_valence() {
	for (int i = 0; i < mesh.getNumVertices(); i++){
		max_valence = max(mesh.getVertex(i).calcValence(), max_valence);
	}
}
void idle(void) {
	if (solve && !selected_vertices.empty()) {
		solve_mesh();
		glutPostRedisplay();
	}
}
void init_rots() {
	int num = mesh.getNumVertices();
	for (int i = 0; i < num; ++i)
	{
		mesh.getVertex(i).setRotation(gsl_matrix_calloc(3,3));
	}
	num = mesh.getNumFaces();
	for (int i = 0; i < num; ++i)
	{
		mesh.getFace(i).setRotation(gsl_matrix_calloc(3,3));
	}
}
int main(int argc, char * argv[])
{
	//taucs_logfile("stdout");
	print_help();

    if (argc < 2) { std::cerr << "Usage: mesh_edit <meshfile.vp>. Exiting...\n"; exit(1); }

    mesh.load(argv[1]);
	//mesh.load("data/horse_simple.vp");
	mesh -= mesh.get_center();
    mesh *= defx / mesh.get_diagonal();
	set_max_valence();
	mesh.update_old_pos();
	//mesh.init_voronoi_areas();
	init_rots();
	calc_weights();
	std::cerr << "Read " << mesh.getNumVertices() << " verts, " << mesh.getNumFaces() << " tris.\n";
    std::cerr <<"Max valence: "<<max_valence<<endl;
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowPosition(100,150);
    glutInitWindowSize(defx, defy);
    glutCreateWindow("Mesh Editing");

    glClearColor(1,1,1,0);
    glClearDepth(10000);
	
//	glCullFace(GL_BACK);
	glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);
//	glEnable(GL_BLEND);

//    glEnableClientState(GL_VERTEX_ARRAY);
//    glEnableClientState(GL_COLOR_ARRAY);

//	glDepthFunc(GL_GREATER);

    glutDisplayFunc(display);
    glutReshapeFunc(resize);
	glutIdleFunc(idle);
	
    glutKeyboardFunc(keyboard);
//	glutSpecialFunc(special_key);
    glutMotionFunc(mousemove);
    glutMouseFunc(mouse);

	glutMainLoop();
    return 0;
}
