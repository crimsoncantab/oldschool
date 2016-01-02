#include "Graph.hpp"

#include "CS207/Util.hpp"

typedef Graph<int, int> GraphType;
typedef GraphType::Node Node;

using namespace std;

int main() {
  GraphType g;

  unsigned num = (1 << 31);
  unsigned gb = (num / (1 << 30)) * 4;
  num /= 16;

  cerr << "Testing memory usage." << endl;
  cerr << "Failure means the program devours memory." << endl;
  cerr << "Add/removing " << gb << "GB of nodes.  This may take a long time..." << endl;
  for (unsigned i = 0; i < num; ++i) {
    Node n1 = g.add_node();
    Node n2 = g.add_node();
    Node n3 = g.add_node();
    Node n4 = g.add_node();
    g.add_edge(n1, n2);
    g.add_edge(n1, n3);
    g.add_edge(n1, n4);
    g.add_edge(n2, n3);
    g.add_edge(n2, n4);
    g.add_edge(n3, n4);

    g.remove_node(n2);
    g.remove_edge(n3, n4);
    g.remove_node(n3);
    g.remove_edge(n1, n4);
    g.remove_node(n1);
    g.remove_node(n4);
    assert(g.stats().node_mem_count_ <= 4);
    assert(g.stats().edge_mem_count_ <= 6);
    if (i % (num / (1 << 4)) == 0) {
      cout << ".";
      cout.flush();
    }
  }
  cout << endl;
  return 0;
}
