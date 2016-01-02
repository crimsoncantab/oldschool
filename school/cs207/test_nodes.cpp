#include "Graph.hpp"
#include "CS207/Util.hpp"

typedef Graph<int, int> GraphType;
typedef GraphType::Node Node;

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

  cerr << "Node is " << sizeof(Node) << " bytes";
  sf_print(sizeof(Node) <= 16);

  sf_print(g.add_node(), "Inserting Node");

  sf_print(g.num_nodes() == 1, "Graph has 1 Node");

  sf_print(g.node(0), "Getting Node");
  Node node = g.node(0);

  sf_print(node.graph() == &g, "Node::graph() works");

  sf_print(node.value() == 0, "Node value conserved");

  sf_print(node.index() == 0, "Index is 0");

  g.remove_node(node);
  sf_print(g.num_nodes() == 0, "Removing Node...");

  sf_print(g.num_nodes() == 0, "Graph has 0 Nodes");

  cerr << "Adding 100 Nodes...";
  for (int k = 0; k < 100; ++k)
    g.add_node(k);
  sf_print(g.num_nodes() == 100);
  sf_print(g.node(0).value() == 0, "Node value conserved (0)");
  sf_print(g.node(99).value() == 99, "Node value conserved (99)");

  // Remove 50 Nodes...
  for (unsigned k = 0; k < 50; ++k) {
    unsigned n = (unsigned) CS207::random(0, g.num_nodes());
    g.remove_node(g.node(n));
  }

  sf_print(g.num_nodes() == 50, "Removed 50 nodes");

  cerr << "Checking Node indices...";
  bool succ = true;
  for (unsigned k = 0; succ && k < 50; ++k) {
    Node node = g.node(k);
    if (node.index() != k)
      succ = false;
  }
  sf_print(succ);

  cerr << "Clearing...";
  g.clear();
  sf_print(g.num_nodes() == 0);

  GraphType g2;
  cerr << "Adding 50 Nodes to G1, G2...";
  for (unsigned k = 0; k < 50; ++k) {
    g.add_node();
    g2.add_node();
  }
  sf_print(g.num_nodes() == 50 && g2.num_nodes() == 50);

  sf_print(g2.node(23) == g2.node(23), "G2-G2 Node comparison ==");
  sf_print(g2.node(23) != g2.node(21), "G2-G2 Node comparison !=");
  sf_print(g2.node(23) != g.node(23), "G2-G1 Node comparison !=");
  sf_print(g2.node(23) != g.node(21), "G2-G1 Node comparison !=");

  sf_print(g.node(23) < g.node(21) || g.node(23) > g.node(21),
           "G1-G1 Node comparison < >");
  sf_print(g.node(23) < g2.node(21) || g.node(23) > g2.node(21),
           "G1-G2 Node comparison < >");

  if (fail_count) {
    std::cerr << "\n" << fail_count
	      << (fail_count > 1 ? " FAILURES" : " FAILURE") << std::endl;
    return 1;
  } else
    return 0;
}
