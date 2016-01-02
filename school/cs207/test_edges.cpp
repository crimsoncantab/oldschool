#include "Graph.hpp"
#include "CS207/Util.hpp"

typedef Graph<int, int> GraphType;
typedef GraphType::Node Node;
typedef GraphType::Edge Edge;

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


int main()
{
  GraphType g;

  cerr << "Edge is " << sizeof(Edge) << " bytes";
  sf_print(sizeof(Edge) <= 24);

  sf_print(g.add_node(), "Inserting Node");
  sf_print(g.add_node(), "Inserting Node");

  sf_print(g.num_nodes() == 2, "Graph has 2 Nodes");

  sf_print(g.add_edge(g.node(0), g.node(1)), "Inserting Edge");

  sf_print(g.edge(0), "Getting Edge");
  Edge e = g.edge(0);

  sf_print(g.num_edges() == 1, "Graph has 1 Edge");

  sf_print(g.has_edge(e.node1(), e.node2()), "Graph has_edge e");

  sf_print((e.node1() == g.node(0) && e.node2() == g.node(1)) ||
           (e.node1() == g.node(1) && e.node2() == g.node(0)),
           "Edge nodes check out");

  auto nr = g.remove_edge(e.node1(), e.node2());
  sf_print(nr == 1, "remove_edge return value");
  sf_print(g.num_edges() == 0, "Removing Edge...");

  sf_print(!g.has_edge(g.node(0), g.node(1)), "Graph !has_edge");

  nr = g.remove_edge(g.node(0), g.node(1));
  sf_print(nr == 0, "remove_edge return value 2");

  sf_print(g.add_edge(g.node(0), g.node(1)), "Inserting Edge");
  sf_print(g.add_edge(g.node(0), g.node(1)), "Inserting Edge Again");

  sf_print(g.num_edges() == 1, "Graph has 1 Edge");

  g.remove_node(g.node(1));
  sf_print(g.num_nodes() == 1, "Removing Node ...");

  sf_print(g.num_edges() == 0, "Edge removed b/c of Node");

  cerr << "Clearing...";
  g.clear();
  sf_print(g.num_nodes() == 0 && g.num_edges() == 0);

  cerr << "Adding 100 Nodes...";
  for (int k = 0; k < 100; ++k) {
    g.add_node();
  }
  sf_print(true);

  // Adding 100 Edges
  for (unsigned k = 0; k < 100; ++k) {
    unsigned n1 = (unsigned) CS207::random(0, g.num_nodes());
    unsigned n2 = (unsigned) CS207::random(0, g.num_nodes());
    while (n1 == n2 || g.has_edge(g.node(n1), g.node(n2))) {
      n1 = (unsigned) CS207::random(0, g.num_nodes());
      n2 = (unsigned) CS207::random(0, g.num_nodes());
    }

    Edge e = g.add_edge(g.node(n1), g.node(n2));
    if (k == 43) {
      sf_print(g.has_edge(e.node1(), e.node2()), "Graph has_edge e");
      sf_print((e.node1() == g.node(n1) && e.node2() == g.node(n2)) ||
               (e.node1() == g.node(n2) && e.node2() == g.node(n1)),
               "Edge nodes check out");
    }
  }

  sf_print(g.num_nodes() == 100 && g.num_edges() == 100, "100 Nodes, 100 Edges");

  // Remove 50 Edges
  for (unsigned k = 0; k < 50; ++k) {
    unsigned n1, n2;
    do {
      n1 = (unsigned) CS207::random(0, g.num_nodes());
      n2 = (unsigned) CS207::random(0, g.num_nodes());
    } while (!g.has_edge(g.node(n1), g.node(n2)));

    g.remove_edge(g.node(n1), g.node(n2));

    if (k == 23)
      sf_print(!g.has_edge(g.node(n1), g.node(n2)),
               "Graph !has_edge after remove");
  }
  sf_print(g.num_edges() == 50, "Removed 50 Edges");

  // Count edges the long way
  unsigned count_edges = 0;
  for (unsigned k = 0; k < g.num_nodes(); ++k) {
    for (unsigned j = k+1; j < g.num_nodes(); ++j) {
      if (g.has_edge(g.node(k), g.node(j)))
        ++count_edges;
    }
  }

  sf_print(count_edges == g.num_edges(), "Edge count agrees");


  // Remove 50 Nodes...
  for (unsigned k = 0; k < 50; ++k) {
    unsigned n = (unsigned) CS207::random(0, g.num_nodes());
    Node node = g.node(n);
    node.value();
    g.remove_node(g.node(n));
  }
  sf_print(g.num_nodes() == 50, "Removed 50 Nodes");

  // Count edges the long way
  count_edges = 0;
  for (unsigned k = 0; k < g.num_nodes(); ++k) {
    for (unsigned j = k+1; j < g.num_nodes(); ++j) {
      if (g.has_edge(g.node(k), g.node(j)))
        ++count_edges;
    }
  }

  sf_print(count_edges == g.num_edges(), "Edge count agrees");

  cerr << "Clearing...";
  g.clear();
  sf_print(g.num_nodes() == 0 && g.num_edges() == 0);

  GraphType g2;
  cerr << "Adding 10 Nodes to G1, G2...";
  for (unsigned k = 0; k < 10; ++k) {
    g.add_node();
    g2.add_node();
  }
  sf_print(true);

  Edge e1 = g.add_edge(g.node(3), g.node(4));
  Edge e2 = g2.add_edge(g2.node(3), g2.node(4));

  sf_print(e1 != e2, "G1-G2 Edge comparison !=");

  if (fail_count) {
    std::cerr << "\n" << fail_count
	      << (fail_count > 1 ? " FAILURES" : " FAILURE") << std::endl;
    return 1;
  } else
    return 0;
}
