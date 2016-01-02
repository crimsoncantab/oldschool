#ifndef MESH_H
#define MESH_H

#include <iostream>
#include <cstdio>
#include <vector>
#include <map>
#include <utility>
#include "vec.h"
//extern "C" {
//	#include <taucs.h>
//}


using namespace std;
class Mesh
{
    typedef int vertex_index;
    typedef int edge_index;
    typedef int face_index;

    struct face_t
    {
        int vertex_[3];
        int edge_[3];
		void * rotation_;
    };
    struct vertex_t
    {
		Vector3 position_;
		Vector3 old_position_;
        Vector3 normal_;
        int halfedge_;
		int valence_;
		void * rotation_;
		float A_;
    };
    struct edge_t
    {
        int halfedge_[2];
    };
    
    std::vector <face_t> face_;
    std::vector <vertex_t> vertex_;
    std::vector <edge_t> edge_;
    
    int fn__(const int i) const														{ return 3; }
    void init_voronoi_areas__()
	{
		const int nt = face_.size(), nv = vertex_.size();
		for (int i = 0; i < nv; ++i) vertex_[i].A_ = 0;
		for (int i = 0; i < nt; i++)
		{
			const Vector3 e[3] = {vertex_[face_[i].vertex_[2]].position_ - vertex_[face_[i].vertex_[1]].position_,
			     					vertex_[face_[i].vertex_[0]].position_ - vertex_[face_[i].vertex_[2]].position_,
			     					vertex_[face_[i].vertex_[1]].position_ - vertex_[face_[i].vertex_[0]].position_ };
			const float area = 0.5f * Vector3::cross(e[0], e[1]).length();
			const float l2[3] = {e[0].length2(), e[1].length2(), e[2].length2()};
			const float ew[3] = {l2[0] * (l2[1] + l2[2] - l2[0]), l2[1] * (l2[2] + l2[0] - l2[1]), l2[2] * (l2[0] + l2[1] - l2[2])};
			Vector3 ca;
			if (ew[0] <= 0)
			{
				ca[1] = -0.25f * l2[2] * area / Vector3::dot(e[0], e[2]);
				ca[2] = -0.25f * l2[1] * area / Vector3::dot(e[0], e[1]);
				ca[0] = area - ca[1] - ca[2];
			} else if (ew[1] <= 0.0f)
			{
				ca[2] = -0.25f * l2[0] * area / Vector3::dot(e[1], e[0]);
				ca[0] = -0.25f * l2[2] * area / Vector3::dot(e[1], e[2]);
				ca[1] = area - ca[2] - ca[0];
			} else if (ew[2] <= 0.0f)
			{
				ca[0] = -0.25f * l2[1] * area / Vector3::dot(e[2], e[1]);
				ca[1] = -0.25f * l2[0] * area / Vector3::dot(e[2], e[0]);
				ca[2] = area - ca[0] - ca[1];
			} else
			{
				const float ewscale = 0.5f * area / (ew[0] + ew[1] + ew[2]);
				for (int j = 0; j < 3; j++) ca[j] = ewscale * (ew[(j+1) % 3] + ew[(j+2) % 3]);
			}
			vertex_[face_[i].vertex_[0]].A_ += ca[0];
			vertex_[face_[i].vertex_[1]].A_ += ca[1];
			vertex_[face_[i].vertex_[2]].A_ += ca[2];
		}
	}

	void init_topology__()
    {
        std::map <std::pair <int, int>, vec_t <int, 2> > E;
        for (std::size_t i = 0; i < face_.size(); ++i)
        {
            const int n = fn__(i);
            for (int j = 0; j < n; ++j)
            {
                const int vj = i | (j<<28);
                const int k = (j+1) % n;
                std::pair <int, int> e(face_[i].vertex_[j], face_[i].vertex_[k]);
                if (e.first < e.second) { const int t = e.first; e.first = e.second; e.second = t; }
                if (E.find(e) == E.end())
                {
                    E[e] = vec_t <int, 2> (vj, -1);
                }
                else
                {
                    vec_t <int, 2>& v = E[e];
                    v[1] = vj;
                }
            }
        }
        edge_.resize(E.size());
        int e = 0;
        for (std::map <std::pair <int, int>, vec_t <int, 2> >::iterator i = E.begin(); i != E.end(); ++i, ++e)
        {
            edge_[e].halfedge_[0] = i->second[0];edge_[e].halfedge_[1] = i->second[1];
            for (int j = 0; j < 2; ++j) if (i->second[j] != -1) face_[i->second[j] & ((1<<28)-1)].edge_[i->second[j] >> 28] = e | (j<<28);
        }
    }
	void load__(const char filename[])
    {
        FILE * const f = std::fopen(filename, "r");
        if (!f) { std::cerr << "Unable to open file: " << filename << std::endl; exit(1); }
        int nv, nt;																			// number of: vertices, tris
        std::fscanf(f, "%d %d", &nv, &nt);
        vertex_.resize(nv);
        face_.resize(nt);
        for (int i = 0; i < nv; ++i) std::fscanf(f, "%lf %lf %lf", &vertex_[i].position_[0], &vertex_[i].position_[1], &vertex_[i].position_[2]);
        for (int i = 0; i < nt; ++i) { std::fscanf(f, "%d %d %d", &face_[i].vertex_[0], &face_[i].vertex_[1], &face_[i].vertex_[2]); face_[i].vertex_[3] = -1; }
        for (int i = 0; i < nt; ++i) for (int j = 0; j < 3; ++j) vertex_[face_[i].vertex_[j]].halfedge_ = i | (j<<28);
        std::fclose(f);
        init_topology__();
    }
public:    
    struct VertexIterator;																		// forward declaration (needed by Vertex class)

    // Default contructor. Assignment operator/constructor
    Mesh()																			{}
    Mesh(const Mesh& m)																{ *this = m; }
    Mesh& operator = (const Mesh& m)												{ face_ = m.face_; vertex_ = m.vertex_; edge_ = m.edge_; /*f_ = m.f_; e_ = m.e_; v_ = m.v_;*/  return *this; }

    // Mesh::Vertex class 
    struct Vertex
    {
        Mesh& m_;
        const int v_;
        
        Vertex(Mesh& m, const int v) : m_(m), v_(v)									{}
        Vector3 getPosition() const													{ return m_.vertex_[v_].position_; }
        Vector3 getOldPosition() const												{ return m_.vertex_[v_].old_position_; }
		void * getRotation() const													{ return m_.vertex_[v_].rotation_; }
        Vector3 getNormal() const													{ assert(m_.vertex_[v_].normal_[0] > -1e37 || !"Error: This normal is uninitialized, you can set it with setNormal()"); return m_.vertex_[v_].normal_; }
		int getValence() const														{ return m_.vertex_[v_].valence_; }
		int getIndex() const														{ return v_; }
        void setPosition(const Vector3& p) const									{ m_.vertex_[v_].position_ = p; }
        void setOldPosition(const Vector3& p) const									{ m_.vertex_[v_].old_position_ = p; }
        void setRotation(void * rotation) const										{ m_.vertex_[v_].rotation_ = rotation; }
        void setNormal(const Vector3& n) const										{ m_.vertex_[v_].normal_ = n; }
		int calcValence() const {
			int my_valence = 0;
			VertexIterator it = getIterator(), it0(it);
			do {
				my_valence++;
				++it;
			} while (it != it0);
			m_.vertex_[v_].valence_ = my_valence;
			return my_valence;
		}
		VertexIterator getIterator() const											{ assert((m_.vertex_[v_].halfedge_&((1<<28)-1)) < m_.face_.size()); return VertexIterator(m_, m_.vertex_[v_].halfedge_); }
    };
    
    // Mesh::Face class
    struct Face
    {
        Mesh& m_;
        const int f_;
        
        Face(Mesh& m, const int f) : m_(m), f_(f)									{}
        int getNumVertices() const													{ return m_.fn__(f_); }
        Vector3 getNormal() const													{ return Vector3::cross(m_.vertex_[m_.face_[f_].vertex_[1]].position_ - m_.vertex_[m_.face_[f_].vertex_[0]].position_,
                                                                                                            m_.vertex_[m_.face_[f_].vertex_[2]].position_ - m_.vertex_[m_.face_[f_].vertex_[0]].position_).normalize(); }
        Vertex getVertex(const int i) const											{ assert(i >= 0 && i < getNumVertices()); return Vertex(m_, m_.face_[f_].vertex_[i]); }
		void * getRotation() const													{ return m_.face_[f_].rotation_; }
        void setRotation(void * rotation) const										{ m_.face_[f_].rotation_ = rotation; }
    };
    
    // Mesh::Edge class
    struct Edge
    {
        Mesh& m_;
        const int e_;

        Edge(Mesh& m, const int e) : m_(m), e_(e)									{}
        Vertex getVertex(const int i) const											{ assert(i >= 0 && i < 2); return Vertex(m_, m_.face_[m_.edge_[e_].halfedge_[0] & ((1<<28)-1)].vertex_[((m_.edge_[e_].halfedge_[0] >> 28) + i) % 3]); }
		Face getFace(const int i) const												{ assert(i >= 0 && i < 2); return Face(m_, m_.edge_[e_].halfedge_[i] & ((1<<28)-1)); }
        bool is_valid() const														{ return getVertex(0).v_ != -1 && getVertex(1).v_ != -1; }
		bool is_boundary() const													{ return m_.edge_[e_].halfedge_[1] == -1; }
    };

    // Mesh::VertexIterator
    struct VertexIterator
    {
        Mesh& m_;
        int h_;
        
        VertexIterator(Mesh& m, const int h) : m_(m), h_(h)							{}
        Vertex getVertex() const													{ const int v(h_ >> 28); const int f(h_ & ((1<<28)-1)); return Vertex(m_, m_.face_[f].vertex_[(v+1) % m_.fn__(f)]); }
        Face getFace() const														{ return Face(m_, h_ & ((1<<28)-1)); }
        VertexIterator& operator ++ ()
        {
            const int f(h_ & ((1<<28)-1)), v(h_ >> 28), vj((v+m_.fn__(f)-1) % m_.fn__(f)), e(m_.face_[f].edge_[vj] & ((1<<28)-1)), ei(m_.face_[f].edge_[vj] >> 28);
            h_ = m_.edge_[e].halfedge_[ei ^ 1];
            return *this;
        }
        bool operator == (const VertexIterator& vi) const							{ return &m_ == &vi.m_ && h_ == vi.h_; }
        bool operator != (const VertexIterator& vi) const							{ return &m_ != &vi.m_ || h_ != vi.h_; }
    };
    
    int getNumFaces() const															{ return face_.size(); }
    int getNumEdges() const															{ return edge_.size(); }
    int getNumVertices() const														{ return vertex_.size(); }

    Vertex getVertex(const int i)													{ return Vertex(*this, i); }
    Edge getEdge(const int i)														{ return Edge(*this, i); }
    Face getFace(const int i)														{ return Face(*this, i); }
	Vector3 get_center() const																						// average of vertex's positions
	{
		Vector3 c(0.0);
		for (std::size_t i = 0; i < vertex_.size(); ++i) for (int j = 0; j < 3; ++j) c[j] += vertex_[i].position_[j];
		c /= vertex_.size();
		return Vector3 (c[0], c[1], c[2]);
	}

	Vector3 get_old_center() const
	{
		Vector3 c(0.0);
		for (std::size_t i = 0; i < vertex_.size(); ++i) for (int j = 0; j < 3; ++j) c[j] += vertex_[i].old_position_[j];
		c /= vertex_.size();
		return Vector3 (c[0], c[1], c[2]);
	}
    void init_normals()
	{
		for (int i = 0; i < getNumVertices(); ++i) getVertex(i).setNormal(Vector3(0));
		for (int i = 0; i < getNumFaces(); ++i)
		{
			const Vector3 n = getFace(i).getNormal();
			if (std::abs(Vector3::dot(n, n) - 1) < 1e-6)										// we only use valid normals
			for (int j = 0; j < getFace(i).getNumVertices(); ++j) getFace(i).getVertex(j).setNormal(getFace(i).getVertex(j).getNormal() + n);
		}
		for (int i = 0; i < getNumVertices(); ++i) getVertex(i).setNormal(getVertex(i).getNormal().length2() < 1e-10 ? Vector3() : getVertex(i).getNormal().normalize());
	}
	void init_voronoi_areas() {
		init_voronoi_areas__();
	}
	float get_diagonal() const
	{
		Vector3 m(1e30f), M(-1e30f);
		for (std::size_t i = 0; i < vertex_.size(); ++i)
		for (int j = 0; j < 3; ++j)
		{
			m[j] = min(m[j], vertex_[i].position_[j]);
			M[j] = max(M[j], vertex_[i].position_[j]);
		}
		return m.dist(M);
	}
	Mesh& operator += (const Vector3& p)								{ for (std::size_t i = 0; i < vertex_.size(); ++i) vertex_[i].position_ += p; return *this; }
	Mesh& operator -= (const Vector3& p)								{ for (std::size_t i = 0; i < vertex_.size(); ++i) vertex_[i].position_ -= p; return *this; }
	Mesh& operator *= (const float a)											{ for (std::size_t i = 0; i < vertex_.size(); ++i) vertex_[i].position_ *= a; return *this; }
	Mesh& operator /= (const float a)											{ for (std::size_t i = 0; i < vertex_.size(); ++i) vertex_[i].position_ /= a; return *this; }
	void load(const char filename[])												{ load__(filename); init_normals();}
	void update_old_pos()												{ for (int i = 0; i < vertex_.size(); ++i) vertex_[i].old_position_ = vertex_[i].position_;}
};

#endif