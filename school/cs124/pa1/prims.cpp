#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
extern "C" {
  #include "gsl/gsl_rng.h"
  #include "gsl/gsl_randist.h"
}

#define PARENT(i) i/2
#define LEFT(i) i*2
#define RIGHT(i) i*2 + 1
#define ROOT 1

struct vert {
	double * coords;
	int heap_index;
	double val;
	vert * prev;
	bool in_s;
};

struct heap {
	vert ** V;
	int size;
};

struct graph {
	vert * V;
	int n;
};

using namespace std;

void swap(heap * h, int a, int b) {
	vert * temp = h->V[b];
	h->V[b] = h->V[a];
	h->V[b]->heap_index = b;
	h->V[a] = temp;
	h->V[a]->heap_index = a;
}

void heapify(heap * h, int n) {
	vert * m = h->V[n];
	vert * l = h->V[LEFT(n)];
	vert * r = h->V[RIGHT(n)];
	int smallest;
	if (LEFT(n) <= h->size && l->val < m->val) {
		smallest = LEFT(n);
	} else {
		smallest = n;
	}
		
	if (RIGHT(n) <= h->size && r->val < h->V[smallest]->val) {
		smallest = RIGHT(n);
	}
	
	if (smallest != n) {
		swap(h, smallest, n);
		heapify(h, smallest);		
	}
}

vert * deletemin(heap * h) {
	vert * min = h->V[ROOT];
	h->V[ROOT] = h->V[h->size];
	h->V[h->size] = NULL;
	h->size--;
	heapify(h, ROOT);
	//min->heap_index = 0;
	return min;
}

void insert(heap * h, vert * v) {
	if (v->heap_index == 0) { //only insert if not in heap
		h->size++;
		h->V[h->size] = v;
		v->heap_index = h->size;
	}
	int n = h->size;
	while (n != ROOT && h->V[PARENT(n)]->val > h->V[n]->val) {
		swap(h, PARENT(n), n);
		n = PARENT(n);
	}
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

void run_prims(graph * g, int dimension, gsl_rng * randgen) {
	heap * h = new heap;
	h->size = 0;
	h->V = new vert*[g->n + 1];
	
	g->V[0].val = 0;
	g->V[0].in_s = true;
	insert(h, &(g->V[0]));
	int counter = 0;
	while (h->size > 0) {
		vert * v = deletemin(h);
		v->in_s = true;
		for (int i = 0; i < g->n; i++) {
			vert * v2 = &(g->V[i]);
			if (!(v2->in_s)) {
				double dist = (dimension > 0) ? calc_eucl_dist(v, v2, dimension): gsl_rng_uniform(randgen);
				//cout <<dist<<endl;
				if ((v2->val < 0. /*initialized to -1*/ || v2->val > dist)) {
					v2->val = dist;
					v2->prev = v;
					insert(h, v2);
					counter++;
				}
			}
		}
	}
	//cout<<"counter: " << counter<<endl;
	delete[] h->V;
	delete h;
}

graph * create_graph(int numvertices, int dimension, gsl_rng * randgen) {
	//create graph
	//using some robust rand function
	vert * vertices = new vert[numvertices];
	for (int i = 0; i < numvertices; i++) {
		vertices[i].heap_index = 0;
		vertices[i].val = -1.;
		vertices[i].in_s = false;
		if (dimension > 0) {
			double * coords = new double[dimension];
			for (int j = 0; j < dimension; j++) {
				coords[j] = gsl_rng_uniform(randgen);
			}
			vertices[i].coords = coords;
		}
	}	
	
	//create graph struct and return
	graph * g = new graph;
	g->V = vertices;
	g->n = numvertices;
	return g;
}

void delete_graph(graph * g, int dim) {
	if (dim > 0) {
		for (int i = 0; i < g->n; i++) {
			delete[] g->V[i].coords;
		}
	}
	delete[] g->V;
	delete g;
}

void print_graph(graph * g, int dimensions) {
	cout << "Vertices:" << endl;
	for (int i = 0; i < g->n; i++) {
		cout<< g->V[i].val << "(";
		for (int j = 0; j < dimensions; j++) {
			cout<<g->V[i].coords[j]<<",";
		}
		cout<<")"<<endl;
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
	double avg = 0.;

	gsl_rng * randgen = gsl_rng_alloc (gsl_rng_mt19937);
	gsl_rng_set(randgen, 50);
	//gsl_rng_set(randgen, time(NULL));
	
	for (int i = 0; i < numtrials; i++ ) {
		clock_t start = clock();
		//create graph
		graph * g = create_graph(numpoints, dimension, randgen);
		
		//run prims's
		run_prims(g, dimension, randgen);
		clock_t end = clock();
		
		print_graph(g, dimension);
		
		double total_w = 0.;
		for (int i = 0; i < g->n; i++) {
			total_w += g->V[i].val;
		}
		
		avg += total_w;
		double sec = ((double) (end - start) ) / CLOCKS_PER_SEC;
		cout << "Tree weight:  " << total_w << endl << "Time: " << sec << " seconds." <<endl;
		delete_graph(g, dimension);
	}
	gsl_rng_free(randgen);

	
	avg /= (double) numtrials;
	
	//print out results
	cout << "number of vertices:  " << numpoints << endl;
	cout << "dimension:  " << dimension << endl;
	cout << "average tree weight:  " << avg << endl;

	return 0;
}
