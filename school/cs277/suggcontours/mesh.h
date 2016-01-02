#ifndef MESH_H
#define MESH_H

#include <iostream>
#include <cstdio>
#include <vector>
#include <map>
#include <utility>
#include "vec.h"



class Mesh
{
    typedef int vertex_index;
    typedef int edge_index;
    typedef int face_index;

    struct face_t
    {
        Vector <int, 4> vertex_;																// this will be either a tri or a quad (face_t::vertex[3] == -1  => this is a tri)
        Vector <int, 4> edge_;
		float dwkr;
    };
    struct vertex_t
    {
        Vector3 position_;
        Vector3 normal_;
		Vector3 e1;
		Vector3 e2;
		Vector3 w;
		Vector3 view;
		float k1;
		float k2;
		float kr;
		float dwkr;
        int halfedge_;
    };
    struct edge_t
    {
        Vector <int, 2> halfedge_;
    };
    
    std::vector <face_t> face_;
    std::vector <vertex_t> vertex_;
    std::vector <edge_t> edge_;
    
    int fn__(const int i) const														{ return face_[i].vertex_[3] == -1 ? 3 : 4; }
    void init_topology__()
    {
        std::map <std::pair <int, int>, Vector <int, 2> > E;
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
                    E[e] = Vector <int, 2> (vj, -1);
                }
                else
                {
                    Vector <int, 2>& v = E[e];
                    v[1] = vj;
                }
            }
        }
        edge_.resize(E.size());
        int e = 0;
        for (std::map <std::pair <int, int>, Vector <int, 2> >::iterator i = E.begin(); i != E.end(); ++i, ++e)
        {
            edge_[e].halfedge_ = i->second;
            for (int j = 0; j < 2; ++j) if (i->second[j] != -1) face_[i->second[j] & ((1<<28)-1)].edge_[i->second[j] >> 28] = e | (j<<28);
        }
    }
    void load__(const char filename[])
    {
        FILE * const f = std::fopen(filename, "r");
        if (!f) { std::cerr << "Unable to open file: " << filename << std::endl; exit(1); }
        int nv, nt, nq;																			// number of: vertices, tris, quads
        std::fscanf(f, "%d %d %d", &nv, &nt, &nq);
        vertex_.resize(nv);
        face_.resize(nt+nq);
        for (int i = 0; i < nv; ++i) std::fscanf(f, "%lf %lf %lf", &vertex_[i].position_[0], &vertex_[i].position_[1], &vertex_[i].position_[2]);
        for (int i = 0; i < nt; ++i) { std::fscanf(f, "%d %d %d", &face_[i].vertex_[0], &face_[i].vertex_[1], &face_[i].vertex_[2]); face_[i].vertex_[3] = -1; }
        for (int i = 0; i < nq; ++i) std::fscanf(f, "%d %d %d %d", &face_[nt+i].vertex_[0], &face_[nt+i].vertex_[1], &face_[nt+i].vertex_[2], &face_[nt+i].vertex_[3]);
        for (int i = 0; i < nt; ++i) for (int j = 0; j < 3; ++j) vertex_[face_[i].vertex_[j]].halfedge_ = i | (j<<28);
        for (int i = 0; i < nq; ++i) for (int j = 0; j < 4; ++j) vertex_[face_[nt+i].vertex_[j]].halfedge_ = i | (j<<28);
        std::fclose(f);
        init_topology__();
        Vector3 center(0);
        for (std::size_t i = 0; i < vertex_.size(); ++i) center += vertex_[i].position_;
        center /= vertex_.size();
        for (std::size_t i = 0; i < vertex_.size(); ++i) vertex_[i].position_ -= center;
        double rms = 0;
        for (std::size_t i = 0; i < vertex_.size(); ++i) rms += Vector3::dot(vertex_[i].position_, vertex_[i].position_);
        rms = std::sqrt(rms / vertex_.size());
        for (std::size_t i = 0; i < vertex_.size(); ++i) vertex_[i].position_ *= 1/rms;
        for (std::size_t i = 0; i < vertex_.size(); ++i) vertex_[i].normal_[0] = -5e37;
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
        Vector3 getNormal() const													{ assert(m_.vertex_[v_].normal_[0] > -1e37 || !"Error: This normal is uninitialized, you can set it with setNormal()"); return m_.vertex_[v_].normal_; }
		Vector3 getE1() const														{ return m_.vertex_[v_].e1; }
		Vector3 getE2() const														{ return m_.vertex_[v_].e2; }
		float getK1() const															{ return m_.vertex_[v_].k1; }
		float getK2() const															{ return m_.vertex_[v_].k2; }
		float getKr() const															{ return m_.vertex_[v_].kr; }
		float getDwKr() const														{ return m_.vertex_[v_].dwkr; }
		Vector3 getW() const														{ return m_.vertex_[v_].w; }
		Vector3 getView() const														{ return m_.vertex_[v_].view; }
        int getIndex() const														{ return v_; }
        void setPosition(const Vector3& p) const									{ m_.vertex_[v_].position_ = p; }
        void setNormal(const Vector3& n) const										{ m_.vertex_[v_].normal_ = n; }
		void setE1(const Vector3& e1) const											{ m_.vertex_[v_].e1 = e1; }
		void setE2(const Vector3& e2) const											{ m_.vertex_[v_].e2 = e2; }
		void setK1(const float k1) const											{ m_.vertex_[v_].k1 = k1; }
		void setK2(const float k2) const											{ m_.vertex_[v_].k2 = k2; }
		void setKr(const float kr) const											{ m_.vertex_[v_].kr = kr; }
		void addDwKr(const float dwkr) const										{ m_.vertex_[v_].dwkr += dwkr; }
		void setDwKr(const float dwkr) const										{ m_.vertex_[v_].dwkr = dwkr; }
		void setW(const Vector3& w) const											{ m_.vertex_[v_].w = w; }
		void setView(const Vector3& view) const										{ m_.vertex_[v_].view = view; }
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
		void setDwKr(const float dwkr) const										{ m_.face_[f_].dwkr = dwkr; }
		float getDwKr() const														{ return m_.face_[f_].dwkr; }
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
    void load(const char filename[])												{ load__(filename); }
};

#endif