#ifndef MESH_H
#define MESH_H



#include <cstdio>
#include <vector>
#include <map>
#include <iostream>
#include <queue>
#include "vec.h"

struct mesh_t
{
	struct vertex_t
	{
		vec_t <float, 3> position_;
		vec_t <float, 3> normal_;
		float A_;																		// its voronoi area
		int triangle_;// one tri that contains the vertex
		int valence;
	};
	struct triangle_t
	{
		int vertex_[3];
		int neighbor_[3];
	};
	typedef int half_edge_t;

	std::vector <vertex_t> vertex_;
	std::vector <triangle_t> triangle_;
	static int dummy_;

private:
	void init_voronoi_areas__()
	{
		const int nt = triangle_.size(), nv = vertex_.size();
		for (int i = 0; i < nv; ++i) vertex_[i].A_ = 0;
		for (int i = 0; i < nt; i++)
		{
			const vec_t <float, 3> e[3] = {vertex_[triangle_[i].vertex_[2]].position_ - vertex_[triangle_[i].vertex_[1]].position_,
			     					vertex_[triangle_[i].vertex_[0]].position_ - vertex_[triangle_[i].vertex_[2]].position_,
			     					vertex_[triangle_[i].vertex_[1]].position_ - vertex_[triangle_[i].vertex_[0]].position_ };
			const float area = 0.5f * vec_t <float, 3>::cross(e[0], e[1]).length();
			const float l2[3] = {e[0].length2(), e[1].length2(), e[2].length2()};
			const float ew[3] = {l2[0] * (l2[1] + l2[2] - l2[0]), l2[1] * (l2[2] + l2[0] - l2[1]), l2[2] * (l2[0] + l2[1] - l2[2])};
			vec_t <float, 3> ca;
			if (ew[0] <= 0)
			{
				ca[1] = -0.25f * l2[2] * area / vec_t <float, 3>::dot(e[0], e[2]);
				ca[2] = -0.25f * l2[1] * area / vec_t <float, 3>::dot(e[0], e[1]);
				ca[0] = area - ca[1] - ca[2];
			} else if (ew[1] <= 0.0f)
			{
				ca[2] = -0.25f * l2[0] * area / vec_t <float, 3>::dot(e[1], e[0]);
				ca[0] = -0.25f * l2[2] * area / vec_t <float, 3>::dot(e[1], e[2]);
				ca[1] = area - ca[2] - ca[0];
			} else if (ew[2] <= 0.0f)
			{
				ca[0] = -0.25f * l2[1] * area / vec_t <float, 3>::dot(e[2], e[1]);
				ca[1] = -0.25f * l2[0] * area / vec_t <float, 3>::dot(e[2], e[0]);
				ca[2] = area - ca[0] - ca[1];
			} else
			{
				const float ewscale = 0.5f * area / (ew[0] + ew[1] + ew[2]);
				for (int j = 0; j < 3; j++) ca[j] = ewscale * (ew[(j+1) % 3] + ew[(j+2) % 3]);
			}
			vertex_[triangle_[i].vertex_[0]].A_ += ca[0];
			vertex_[triangle_[i].vertex_[1]].A_ += ca[1];
			vertex_[triangle_[i].vertex_[2]].A_ += ca[2];
		}
	}

	void init_topology__()														// initializes vertex_t's triangle_ field and triangle_t's neighbor_ field
	{
		std::map <std::pair <int, int>, std::pair <int, int> > e;
		for (std::size_t i = 0; i < triangle_.size(); ++i) for (int j = 2; j >= 0; --j) triangle_[i].neighbor_[j] = -1;
		for (std::size_t i = 0; i < triangle_.size(); ++i) for (int j = 2; j >= 0; --j)
		{
			const int k = (j+1) % 3;
			std::pair <int, int> e__(triangle_[i].vertex_[j], triangle_[i].vertex_[k]);
			if (e__.first > e__.second) { const int t(e__.first); e__.first = e__.second; e__.second = t; }
			if (e.find(e__) == e.end())
			{
				e[e__] = std::make_pair(i | (j << 28), -1);
			}
			else
			{
				if (e[e__].second != -1) { std::cerr << "Non-manifold mesh. Exiting...\n"; exit(1); }
				e[e__].second = i | (j << 28);
			}
		}
		for (std::map <std::pair <int, int>, std::pair <int, int> >::iterator it = e.begin(); it != e.end(); ++it)
		{
			triangle_[it->second.first & ((1 << 28)-1)].neighbor_[it->second.first >> 28] = it->second.second;
			if (it->second.second != -1) triangle_[it->second.second & ((1 << 28)-1)].neighbor_[it->second.second >> 28] = it->second.first;
		}
		bool boundary = false;
		for (std::map <std::pair <int, int>, std::pair <int, int> >::iterator it = e.begin(); !boundary && it != e.end(); ++it) boundary = it->second.second == -1 || it->second.first == -1;
		if (boundary) std::cerr << "Has boundary.\n";
		
		for (std::size_t i = 0; i < vertex_.size(); ++i) vertex_[i].triangle_ = -1;
		for (std::size_t i = 0; i < triangle_.size(); ++i) for (int j = 2; j >= 0; --j) vertex_[triangle_[i].vertex_[j]].triangle_ = i | (j << 28);
	}

	static half_edge_t half_edge_inc__(const half_edge_t h)						{ return (h & ((1<<28) - 1)) | ((((h >> 28)+1) % 3) << 28); }
	static half_edge_t half_edge_dec__(const half_edge_t h)						{ return (h & ((1<<28) - 1)) | ((((h >> 28)+2) % 3) << 28); }

public:
	mesh_t() 																	{}
	
	half_edge_t begin(const int i) const
	{
		const half_edge_t r = vertex_[i].triangle_;
		assert(triangle_[r & ((1<<28) - 1)].vertex_[r >> 28] == i);
		return r;
	}
	half_edge_t increment(const half_edge_t he) const
	{
		return half_edge_inc__(triangle_[he & ((1<<28)-1)].neighbor_[he >> 28]);
	}
	half_edge_t decrement(const half_edge_t he) const
	{
		const half_edge_t h = half_edge_dec__(he);
		return triangle_[h & ((1<<28)-1)].neighbor_[h >> 28];
	}
	int get_incident_vertex(const half_edge_t h) const							{ return ((h>>28) + 1) % 3; }
	int get_incident_triangle(const half_edge_t h) const						{ return h & ((1<<28)-1); }

	int get_num_vertices() const												{ return vertex_.size(); }
	int get_num_triangles() const												{ return triangle_.size(); }
	
	float get_vertex_voronoi_area(const int i) const							{ return vertex_[i].A_; }					// returns vertex i's voronoi-area
	vec_t <float, 3>& get_vertex(const int i)									{ return vertex_[i].position_; }
	vec_t <float, 3>& get_normal(const int i)									{ return vertex_[i].normal_; }
	const vec_t <float, 3>& get_vertex(const int i) const						{ return vertex_[i].position_; }
	const vec_t <float, 3>& get_normal(const int i) const						{ return vertex_[i].normal_; }

	int get_triangle(const int i, const int j) const							{ return triangle_[i].vertex_[j]; }
	int get_neighbor(const int t, const int i, int& j = dummy_) const			{ const int n = triangle_[t].neighbor_[i]; j = n >> 28; return n & ((1 << 28) - 1); }
	bool has_neighbor(const int t, const int i) const							{ return triangle_[t].neighbor_[i] != -1; }
	vec_t <float, 3> get_triangle_normal(const int i) const																	// returns *unnormalized* triangle normal
	{
		const vec_t <float, 3>& p0 = vertex_[triangle_[i].vertex_[0]].position_;
		const vec_t <float, 3>& p1 = vertex_[triangle_[i].vertex_[1]].position_;
		const vec_t <float, 3>& p2 = vertex_[triangle_[i].vertex_[2]].position_;
		return vec_t <float, 3>::cross(p1-p0, p2-p0);
	}
	float get_triangle_area(const int i) const									{ return get_triangle_normal(i).length(); }
	
	vec_t <float, 3> get_center() const																						// average of vertex's positions
	{
		vec_t <double, 3> c(0.0);
		for (std::size_t i = 0; i < vertex_.size(); ++i) for (int j = 0; j < 3; ++j) c[j] += vertex_[i].position_[j];
		c /= vertex_.size();
		return vec_t <float, 3> (c[0], c[1], c[2]);
	}
	mesh_t& operator += (const vec_t <float, 3>& p)								{ for (std::size_t i = 0; i < vertex_.size(); ++i) vertex_[i].position_ += p; return *this; }
	mesh_t& operator -= (const vec_t <float, 3>& p)								{ for (std::size_t i = 0; i < vertex_.size(); ++i) vertex_[i].position_ -= p; return *this; }
	mesh_t& operator *= (const float a)											{ for (std::size_t i = 0; i < vertex_.size(); ++i) vertex_[i].position_ *= a; return *this; }
	mesh_t& operator /= (const float a)											{ for (std::size_t i = 0; i < vertex_.size(); ++i) vertex_[i].position_ /= a; return *this; }
	
	float get_rms() const
	{
		double r(0);
		for (std::size_t i = 0; i < vertex_.size(); ++i) r += vertex_[i].position_.length2();
		return std::sqrt(r / (double)vertex_.size());
	}
	float get_diagonal() const
	{
		vec_t <float, 3> m(1e30f), M(-1e30f);
		for (std::size_t i = 0; i < vertex_.size(); ++i) for (int j = 0; j < 3; ++j)
		{
			m[j] = min(m[j], vertex_[i].position_[j]);
			M[j] = max(M[j], vertex_[i].position_[j]);
		}
		return m.dist(M);
	}
	
	void init_normals()
	{
		for (std::size_t i = 0; i < vertex_.size(); ++i) vertex_[i].normal_ = vec_t <float, 3> (0.f);
		for (std::size_t i = 0; i < triangle_.size(); ++i)
		for (int j = 0; j < 3; ++j)
		{
			const vec_t <float, 3> tn = get_triangle_normal(i);
			vec_t <float, 3>& n = vertex_[triangle_[i].vertex_[j]].normal_;
			if (vec_t <float, 3>::dot(n, tn) < 0) n -= tn; else n += tn;
		}
		for (std::size_t i = 0; i < vertex_.size(); ++i) vertex_[i].normal_.normalize();
		
		init_voronoi_areas__();
	}
	void write_vp(const char filename[])										{ FILE * fp = fopen(filename, "r"); write_vp(fp); fclose(fp); }
	void write_vp(FILE * fp)
	{
		const int nv = vertex_.size();
		const int nt = triangle_.size();
		fprintf(fp, "%d %d\n", nv, nt);
		for (int i = 0; i < nv; ++i) fprintf(fp, "%f %f %f\n", vertex_[i].position_[0], vertex_[i].position_[1], vertex_[i].position_[2]);
		for (int i = 0; i < nt; ++i) fprintf(fp, "%d %d %d\n", triangle_[i].vertex_[0], triangle_[i].vertex_[1], triangle_[i].vertex_[2]);
	}
	void read_vp(const char filename[])											{ FILE * fp = fopen(filename, "r"); read_vp(fp); fclose(fp); }
	void read_vp(FILE * fp)																// reads in simple (vertex-list, triangle-list, quad-list) format
	{
		int nv, nt;
		std::fscanf(fp, "%d %d", &nv, &nt);
		vertex_ = std::vector <vertex_t> (nv);
		triangle_ = std::vector <triangle_t> (nt);
		for (int j = 0; j < nv; ++j) std::fscanf(fp, "%f %f %f", &vertex_[j].position_[0], &vertex_[j].position_[1], &vertex_[j].position_[2]);
		for (int j = 0; j < nt; ++j) std::fscanf(fp, "%d %d %d", &triangle_[j].vertex_[0], &triangle_[j].vertex_[1], &triangle_[j].vertex_[2]);
		init_normals();
		init_topology__();
	}
};




#endif
