#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
extern "C" {
  #include <gsl/gsl_rng.h>
  #include <gsl/gsl_randist.h>
}


struct edge {
	int u;
	int v;
	double weight;
};

struct vert {
	double * coords;
	int parent;
	int rank;
};

struct graph {
	edge ** edges;
	vert * vertices;
	int n;
	int m;
};

using namespace std;

int find(int x, vert * vertices) {
	if (vertices[x].parent != x)
		vertices[x].parent = find(vertices[x].parent, vertices);
	return vertices[x].parent;
}

void makeset(vert * vertices, int n) {
	for (int i = 0; i < n; i++) {
		vertices[i].parent = i;
		vertices[i].rank = 0;
	}
}

int link(int x, int y, vert * vertices) {
	if (vertices[x].rank > vertices[y].rank)  {
		int temp = x;
		x = y;
		y = temp;
	} else if (vertices[x].rank == vertices[y].rank) {
		vertices[y].rank++;
	}
	vertices[x].parent = y;
	return y;
}

void union_lists(int x, int y, vert * vertices) {
	link(find(x, vertices), find(y, vertices), vertices);
}


int comp_edges(const void * elem1, const void * elem2) {
	const edge * e1 = *(edge ** ) elem1;
	const edge * e2 = *(edge ** ) elem2;
	double diff = e1->weight - e2->weight;
//	cout << "e1 " << e1->weight << endl;
//	cout << "e2 " << e2->weight << endl;
//	cout << diff << endl;
	if (diff < 0.) return -1;
	if (diff > 0.) return 1;
	return 0;
}

void sort_edges(edge ** list, int len) {
	qsort(list, len, sizeof(edge *), comp_edges);
}


edge ** run_kruskals(graph * g) {
	
	edge ** mst = new edge*[g->n - 1];
	int treesize = 0;
	
	//sort edges
	sort_edges(g->edges, g->m);
	
	
	vert * vertices = g->vertices;
	edge ** edges = g->edges;
	makeset(vertices, g->n);
	
	for (int i = 0; i < g->m; i ++) {
		edge * e = edges[i];
		if (find(e->u, vertices) != find(e->v, vertices)) {
			mst[treesize] = e;
			treesize++;
			//if (treesize == g->n) break; //we've found the tree
			union_lists(e->u, e->v, g->vertices);
		}
	}
	return mst;	
}

double calc_eucl_dist(vert * vert1, vert * vert2, int dimension) {
	double sum = 0.;
	double * coords1 = vert1->coords;
	double * coords2 = vert2->coords;
	for (int i = 0; i < dimension; i ++) {
		double diff = coords1[i] - coords2[i];
		sum += diff * diff;
	}
	return sqrt(sum);
}

graph * create_graph(int numvertices, int dimension, gsl_rng * randgen) {
	//create graph
	//using some robust rand function
	int numedges = (numvertices * (numvertices - 1)) / 2;
	vert * vertices = new vert[numvertices];
	edge ** edges = new edge*[numedges];
	
	for (int i = 0; i < numvertices; i++) {
		vertices[i].parent = i;
		vertices[i].rank = 0;
		if (dimension > 0) {
			double * coords = new double[dimension];
			for (int j = 0; j < dimension; j++) {
				coords[j] = gsl_rng_uniform(randgen);
			}
			vertices[i].coords = coords;
		}
	}
	int count = 0;
	for (int i = 0; i < numvertices - 1; i++) {
		for (int j = i+1; j < numvertices; j++) {
			edge * e = new edge;
			e->u = i;
			e->v = j;
			if (dimension > 0) {
				e->weight = calc_eucl_dist(&vertices[i], &vertices[j], dimension);
			} else {
				e->weight = gsl_rng_uniform(randgen);
			}
			edges[count] = e;
			count++;
		}
	}
	
	
	//cleanup 
	if (dimension > 0) {
		for (int i = 0; i < numvertices; i++) {
			delete [] vertices[i].coords;
		}
	}
	//create graph struct and return
	graph * g = new graph;
	g->edges = edges;
	g->vertices = vertices;
	g->n = numvertices;
	g->m = numedges;
	return g;
}

void delete_graph(graph * g) {
	delete[] g->vertices;
	edge ** edges = g->edges;
	for (int i = 0; i < g->m; i++) {
		delete edges[i];
	}
	delete[] edges;
	delete g;
}

void print_graph(graph * g) {
	cout << "Edges:" << endl;
	for (int i = 0; i < g->m; i++) {
		cout << "(" << g->edges[i]->u << "," << g->edges[i]->v << ") weight: " << g->edges[i]->weight << endl;
	}
}

int main(int argc, char * argv[]) {
	int numpoints, numtrials, dimension;
	
	if (argc != 5) {
		cout << "Usage: "<< argv[0] << " 0 numpoints numtrials dimension" << endl;
		return 1;
	}
	
	//check command args
	numpoints = atoi(argv[2]);
	numtrials = atoi(argv[3]);
	dimension = atoi(argv[4]);
	if ((numpoints < 0) || (numtrials < 0) ||	(dimension < 0)) {
		cout << "All arguments must be non-negative integers" << endl;
		return 1;
	}

	gsl_rng * randgen = gsl_rng_alloc (gsl_rng_mt19937);
	//gsl_rng_set(randgen, time(NULL));
	gsl_rng_set(randgen,50);
	double avg = 0.;
	for (int i = 0; i < numtrials; i++ ) {	
		//create graph
		graph * g = create_graph(numpoints, dimension, randgen);

		//run kruskal's
		edge ** mst = run_kruskals(g);
	
		print_graph(g);
		
		double total_w = 0.;
		for (int i = 0; i < numpoints-1; i++) {
			total_w += mst[i]->weight;
			cout << mst[i]->weight <<endl;
		}
		
		avg += total_w;
		
		cout << "Tree weight:  " << total_w << endl;
		delete_graph(g);
	}
	gsl_rng_free(randgen);
	
	
	avg /= (double) numtrials;
	
	//print out results
	cout << "number of vertices:  " << numpoints << endl;
	cout << "dimension:  " << dimension << endl;
	cout << "average tree weight:  " << avg << endl;

	return 0;
}
