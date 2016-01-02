/**
 * @file cs207viewer.cpp
 * Test script for the SDLViewer and Graph
 *
 * @brief Reads in two files specified on the command line.
 * First file: 3D Points (one per line) defined by three doubles
 * Second file: Tetrahedra (one per line) defined by 4 indices into the point
 * list
 *
 * Prints
 * A B
 * where A = number of nodes
 *       B = number of edges
 * and launches an SDLViewer to visualize the system
 */

#include "CS207/SDLViewer.hpp"
#include "CS207/Util.hpp"
#include "SpatialGraph.hpp"
#include "Simplex.hpp"
#include "Color.hpp"
#include <fstream>
#include <cstdio>
#include <map>
#include <deque>

/** Construct an induced subgraph of Graph @a g.
 * @param[in] g initial graph
 * @param[in] node_pred Node predicate to determine nodes in @a subgraph
 * @param[out] subgraph The subgraph of @a g induced by @a keep
 *
 * If @a node_pred(n) returns true for a node n of Graph @a g, then a node
 * with that position is added to @a subgraph. All induced edges are edges of
 * @a subgraph. */
template <typename G, typename Predicate>
void induced_subgraph(const G& g, Predicate node_pred, G& subgraph) {
  std::map<int, int> idx2subidx;

  for (typename G::node_iterator n_it = g.node_begin(); n_it < g.node_end(); ++n_it) {
    typename G::Node n = *n_it;
    if (node_pred(*n_it)) {
      idx2subidx[n.index()] = subgraph.add_node(n.position(), n.value()).index();
    }
  }

  for (typename G::edge_iterator e_it = g.edge_begin(); e_it < g.edge_end(); ++e_it) {
    typename G::Edge e = *e_it;
    std::map<int, int>::iterator m_it1 = idx2subidx.find(e.node1().index());
    std::map<int, int>::iterator m_it2 = idx2subidx.find(e.node2().index());
    if (m_it1 != idx2subidx.end() && m_it2 != idx2subidx.end()) {
      assert(m_it1 != m_it2);
      subgraph.add_edge(subgraph.node(m_it1->second), subgraph.node(m_it2->second));
    }
  }
}

/** Test predicate for HW1 #5 */
struct IndexPredicate {
  unsigned last_index;

  IndexPredicate(unsigned last_index)
    : last_index(last_index) {
  }

  template <typename N>
    bool operator()(const N & n) {
    return n.index() < last_index;
  }
};

/** Predicate that returns true for nodes less than a given distance away
 * from a given point. */
struct EuclidDistancePredicate {
  Point p_;
  double distance_;

  EuclidDistancePredicate(Point p, double distance)
    : p_(p), distance_(distance) {
  }

  template <typename N>
    bool operator()(const N & n) {
    return (p_ - n.position()).length() < distance_;
  }
};

/** Test function for HW1 #5 */
template <typename G>
void test_induced_subgraph(const G& graph, CS207::SDLViewer& viewer) {
  G subgraph;
  double distance;
  std::cout << "Pick a distance: ";
  std::cin >> distance;
  std::cout << std::endl;
  EuclidDistancePredicate ep(Point(), distance);
  induced_subgraph(graph, ep, subgraph);

  viewer.clear();
  auto node_map = viewer.empty_node_map(graph);
  viewer.add_nodes(subgraph.node_begin(), subgraph.node_end(), node_map);
  viewer.add_edges(subgraph.edge_begin(), subgraph.edge_end(), node_map);
  viewer.center_view();
}

struct ShortestPathColorFunctor {
  float dist_max_;

  ShortestPathColorFunctor(int dist_max)
    : dist_max_(float(dist_max)) {
  }

  // Member Data

  template <typename NODE >
    Color operator()(const NODE & node) {
    return Color::make_heat(1.0 - (node.value() / dist_max_));
  }
};

/** Calculate shortest path lengths in @a g from @a root.
 * @param[in,out] g input graph
 * @param[in] root root node
 * @return maximum path length in the Graph
 *
 * Changes each node's value() to the length of the shortest path to that node
 * from @a root. @a root's value() is 0. Nodes unreachable from @a root should
 * have value() -1. */
template <typename G>
int shortest_path_lengths(G& g, typename G::Node root) {
  std::deque<typename G::Node> q;
  int dist = 0;

  //initialized all nodes to dist 0
  for (typename G::node_iterator n_it = g.node_begin(); n_it < g.node_end(); ++n_it) {
    (*n_it).value() = dist;
  }

  q.push_back(root);
  while (!q.empty()) {
    typename G::Node n = q.front();
    q.pop_front();
    for (typename G::incident_iterator i_it = n.edge_begin(); i_it < n.edge_end(); ++i_it) {
      typename G::Node n2 = (*i_it).node2();
      if (n2.value() == 0) {
        n2.value() = n.value() + 1;
        q.push_back(n2);
      }
    }
    dist = n.value();
  }
  return dist;
}

int main(int argc, char* argv[]) {
  // check arguments
  if (argc < 2) {
    std::cerr << "Usage: cs207viewer NODES_FILE TETS_FILE\n";
    exit(1);
  }

  SpatialGraph<int, int> graph;

  // Read all Points and add them to the Graph
  std::ifstream nodes_file(argv[1]);
  Point p;
  while (CS207::getline_parsed(nodes_file, p))
    graph.add_node(p);

  // Read all Tetrahedra and add the edges to the Graph
  std::ifstream tets_file(argv[2]);
  Tetrahedron t;
  while (CS207::getline_parsed(tets_file, t))
    if (t.n[0] < graph.size() && t.n[1] < graph.size()
      && t.n[2] < graph.size() && t.n[3] < graph.size()) {
      graph.add_edge(graph.node(t.n[0]), graph.node(t.n[1]));
      graph.add_edge(graph.node(t.n[0]), graph.node(t.n[2]));
      graph.add_edge(graph.node(t.n[0]), graph.node(t.n[3]));
      graph.add_edge(graph.node(t.n[1]), graph.node(t.n[2]));
      graph.add_edge(graph.node(t.n[1]), graph.node(t.n[3]));
      graph.add_edge(graph.node(t.n[2]), graph.node(t.n[3]));
    }

  // Print out the stats
  std::cout << graph.num_nodes() << " " << graph.num_edges() << std::endl;

  int max_dist = shortest_path_lengths(graph, graph.node(0));

  // Launch the SDLViewer
  CS207::SDLViewer viewer;

  viewer.launch();
  //  viewer.draw_graph(graph);
  //test_induced_subgraph(graph, viewer);
  auto node_map = viewer.empty_node_map(graph);
  viewer.add_nodes(graph.node_begin(), graph.node_end(), ShortestPathColorFunctor(max_dist), node_map);
  viewer.add_edges(graph.edge_begin(), graph.edge_end(), node_map);

  return 0;
}
