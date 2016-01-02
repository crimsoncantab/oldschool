#include "Mesh.hpp"
#include "CS207/Util.hpp"

typedef Mesh<int, int, int> MeshType;
typedef MeshType::Node Node;
typedef MeshType::Edge Edge;
typedef MeshType::Triangle Triangle;

using namespace std;
using namespace CS207;
static unsigned fail_count = 0;

template <typename T>
void sf_print(T a, string msg = "") {
  (void) a;
  cerr << msg << " [Success]" << endl;
}

void sf_print(bool sf, string msg = "") {
  if (sf)
    cerr << msg << " [Success]" << endl;
  else {
    cerr << msg << " [FAIL]" << endl;
    ++fail_count;
  }
}

int main() {
  MeshType g;

  Point p1(CS207::random(), CS207::random(), CS207::random());
  Point p2(CS207::random(), CS207::random(), CS207::random());
  Point p3(CS207::random(), CS207::random(), CS207::random());

  //cerr << "Edge is " << sizeof(Edge) << " bytes";
  //sf_print(sizeof(Edge) <= 24);

  sf_print(g.add_node(p1), "Inserting Node");
  sf_print(g.add_node(p2), "Inserting Node");

  sf_print(g.num_nodes() == 2, "Mesh has 2 Nodes");

  sf_print(g.add_node(p3), "Inserting Node");

  sf_print(g.add_triangle(g.node(0), g.node(1), g.node(2)), "Inserting Triangle");

  sf_print(g.edge(0), "Getting Triangle");
  Triangle t = g.triangle(0);

  sf_print(g.num_triangles() == 1, "Graph has 1 Triangle");

  sf_print(g.has_triangle(g.node(0), g.node(1), g.node(2)), "Graph has triangle");


  sf_print((t.node(0) == g.node(0) && t.node(1) == g.node(1) && t.node(2) == g.node(2)) ||
    (t.node(0) == g.node(0) && t.node(1) == g.node(2) && t.node(2) == g.node(1)) ||
    (t.node(0) == g.node(1) && t.node(1) == g.node(0) && t.node(2) == g.node(2)) ||
    (t.node(0) == g.node(1) && t.node(1) == g.node(2) && t.node(2) == g.node(0)) ||
    (t.node(0) == g.node(2) && t.node(1) == g.node(0) && t.node(2) == g.node(1)) ||
    (t.node(0) == g.node(2) && t.node(1) == g.node(1) && t.node(2) == g.node(0)),
    "Triangle Nodes check out");

  Point p4(CS207::random(), CS207::random(), CS207::random());

  sf_print(g.add_node(p4), "Inserting Node");

  sf_print(g.add_triangle(g.node(1), g.node(2), g.node(3)), "Inserting New Triangle Adjacent");

  sf_print(g.num_triangles() == 2, "Graph has 2 Triangles");

  cerr << "Clearing...";
  g.clear();
  sf_print(g.num_triangles() == 0, "Removed All Triangles");

  cerr << "Adding 100 Nodes...";
  for (int k = 0; k < 100; ++k) {
    g.add_node(Point(CS207::random(), CS207::random(), CS207::random()));
  }
  sf_print(true);

  // Adding 100 Triangles
  for (unsigned k = 0; k < 100; ++k) {
    unsigned n1 = (unsigned) CS207::random(0, g.num_nodes());
    unsigned n2 = (unsigned) CS207::random(0, g.num_nodes());
    unsigned n3 = (unsigned) CS207::random(0, g.num_nodes());
    
    while (n1 == n2 || n2 == n3 || n1 == n3 || g.has_triangle(g.node(n1), g.node(n2), g.node(n3))) {
      n1 = (unsigned) CS207::random(0, g.num_nodes());
      n2 = (unsigned) CS207::random(0, g.num_nodes());
      n3 = (unsigned) CS207::random(0, g.num_nodes());
    }
    Triangle t1 = g.add_triangle(g.node(n1), g.node(n2), g.node(n3));
    if (k == 43) {
      sf_print(g.has_triangle(g.node(n1), g.node(n2), g.node(n3)), "Graph has triangle t1");

      sf_print((t1.node(0) == g.node(n1) && t1.node(1) == g.node(n2) && t1.node(2) == g.node(n3)) ||
               (t1.node(0) == g.node(n1) && t1.node(1) == g.node(n3) && t1.node(2) == g.node(n2)) ||
   	       (t1.node(0) == g.node(n2) && t1.node(1) == g.node(n1) && t1.node(2) == g.node(n3)) ||
               (t1.node(0) == g.node(n2) && t1.node(1) == g.node(n3) && t1.node(2) == g.node(n1)) ||
               (t1.node(0) == g.node(n3) && t1.node(1) == g.node(n1) && t1.node(2) == g.node(n2)) ||
               (t1.node(0) == g.node(n3) && t1.node(1) == g.node(n2) && t1.node(2) == g.node(n1)),
        "Triangle Nodes check out");
    }
  }

  sf_print(g.num_nodes() == 100 && g.num_triangles() == 100, "100 Nodes, 100 Triangles");


  // Count edges the long way
  unsigned count_triangles = 0;
  for (unsigned k = 0; k < g.num_nodes(); ++k) {
    for (unsigned j = k + 1; j < g.num_nodes(); ++j) {
      for (unsigned l = j + 1; l < g.num_nodes(); ++l) {
        if (g.has_triangle(g.node(k), g.node(j), g.node(l)))
          ++count_triangles;
      }
    }
  }

  sf_print(count_triangles == g.num_triangles(), "Triangles count agrees");


  cerr << "Clearing...";
  g.clear();
  sf_print(g.num_nodes() == 0 && g.num_triangles() == 0 && g.num_edges() == 0);

  MeshType g2;
  cerr << "Adding 10 Nodes to G1, G2...";
  for (unsigned k = 0; k < 10; ++k) {
    Point p(CS207::random(), CS207::random(), CS207::random());
    g.add_node(p);
    g2.add_node(p);
  }
  sf_print(true);

  Triangle t1 = g.add_triangle(g.node(3), g.node(4), g.node(5));
  Triangle t2 = g2.add_triangle(g2.node(3), g2.node(4), g2.node(5));

  sf_print(t1 != t2, "G1-G2 Triangle comparison !=");

  cerr << "Clearing...";
  g.clear();
  sf_print(g.num_nodes() == 0 && g.num_triangles() == 0 && g.num_edges() == 0);  

  cerr << "Adding 100 Nodes..\n";
  for (int k = 0; k < 100; ++k) {
    g.add_node(Point(CS207::random(), CS207::random(), CS207::random()));
  }

  cerr << "Adding 100 Triangles...\n";

  //Add 100 nodes and 100 triangles for iteration test
  for (unsigned k = 0; k < 100; ++k) {
    unsigned n1 = (unsigned) CS207::random(0, g.num_nodes());
    unsigned n2 = (unsigned) CS207::random(0, g.num_nodes());
    unsigned n3 = (unsigned) CS207::random(0, g.num_nodes());
    
    while (n1 == n2 || n2 == n3 || n1 == n3 || g.has_triangle(g.node(n1), g.node(n2), g.node(n3))) {
      n1 = (unsigned) CS207::random(0, g.num_nodes());
      n2 = (unsigned) CS207::random(0, g.num_nodes());
      n3 = (unsigned) CS207::random(0, g.num_nodes());
    }

    g.add_triangle(g.node(n1), g.node(n2), g.node(n3));
  }
  

  //node iterator test
  unsigned int node_mark[g.num_nodes()];
  bool allMet = true;
  for(unsigned int i=0; i < g.num_nodes(); ++i)
    node_mark[i] = 0;
  for(auto it = g.node_begin(); it != g.node_end(); ++it) {
    Node n = *it;
    node_mark[n.index()] = 1;
  }
  for(unsigned int i=0; i < g.num_nodes(); ++i) {
    if(node_mark[i] != 1)
      allMet = false;
  }

  sf_print(allMet, "Node iterator iterates all nodes");

  //triangle iterator test
  unsigned int triangle_mark[g.num_triangles()];
  allMet = true;
  for(unsigned int i=0; i < g.num_triangles(); ++i)
    triangle_mark[i] = 0;
  for(auto it = g.triangle_begin(); it != g.triangle_end(); ++it) {
    Triangle t3 = *it;
    triangle_mark[t3.index()] = 1;
  }
  for(unsigned int i=0; i < g.num_triangles(); ++i) {
    if(triangle_mark[i] != 1)
      allMet = false;
  }

  sf_print(allMet, "Triangle iterator iterates all triangle");

  //check that all triangle neighbors iterate through
  unsigned int numAdjacent = 0;
  allMet = true;
  bool isAdjacent = true;
  for(auto it = g.triangle_begin(); it < g.triangle_end(); ++it) {
    Triangle t3 = *it;
    
    for(auto it2 = t3.triangle_begin(); it2 != t3.triangle_end(); ++it2) {
      Triangle t3_adj = *it2;
      //test for correct adjacency
      if(!t3.is_adjacent(t3_adj)) {
        isAdjacent = false;
      }
    }
    numAdjacent = 0;
    for(auto it3 = g.triangle_begin(); it3 != g.triangle_end(); ++it3) {
      if(t3.is_adjacent(*it3))
        numAdjacent++;
    }
    if(numAdjacent != t3.num_neighbors())
      allMet = false;

  }
  sf_print(isAdjacent, "All triangle neighbors are adjacent");
  sf_print(allMet, "All triangle neighbors are iterated over");


  //check incident triangle iterator
  int numIncident = 0;
  allMet = true;
  for(auto it_node = g.node_begin(); it_node != g.node_end(); ++it_node) {
    Node n = *it_node;
    numIncident = 0;
    for(auto it_triangle = g.triangle_begin(); it_triangle != g.triangle_end(); ++it_triangle) {
      Triangle tri = *it_triangle;
      if(tri.node(0) == n || tri.node(1) == n || tri.node(2) == n)
        numIncident++;
    }
    for(auto it_triangle_incident = n.triangle_begin(); it_triangle_incident != n.triangle_end(); ++it_triangle_incident) {
      Triangle tri = *it_triangle_incident;
      if(tri.node(0) == n || tri.node(1) == n || tri.node(2) == n)
        numIncident--;
    }
    if(numIncident != 0)
      allMet = false;
  }

  sf_print(allMet, "All triangles incident to a node are iterated over");

  if (fail_count) {
    std::cerr << "\n" << fail_count
      << (fail_count > 1 ? " FAILURES" : " FAILURE") << std::endl;
    return 1;
  } else
    return 0;
}
