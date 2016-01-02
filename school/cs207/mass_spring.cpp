/**
 * @file mass_spring.cpp
 * Implementation of mass-spring system using Graph
 *
 * @brief Reads in two files specified on the command line.
 * First file: 3D Points (one per line) defined by three doubles
 * Second file: Tetrahedra (one per line) defined by 4 indices into the point
 * list
 */

#include "CS207/SDLViewer.hpp"
#include "CS207/Util.hpp"
#include "SpatialGraph.hpp"
#include "Constraint.hpp"
#include "Force.hpp"
#include "Point.hpp"
#include "Tetrahedron.hpp"
#include "Color.hpp"
#include "BoundingBox.hpp"
#include "mass_spring_types.hpp"
#include <fstream>

static const double spring_k = 100;
typedef SpatialGraph<node_state, edge_state, 5 > GraphType;
typedef GraphType::Node Node;
typedef GraphType::Edge Edge;

/** Change a graph's nodes according to a step of the symplectic Euler
 *    method with the given node force.
 * @param[in,out] g graph
 * @param[in] t the current time (useful for time-dependent forces)
 * @param[in] dt the time step
 * @param[in] force function object defining the force per node
 * @param[in] constraint function object defining the constraint on the graph
 * @pre G::node_value_type supports .v_ and .m_
 * @return the next time step (usually @a t + @a dt)
 *
 * @a force is called as @a force(n, @a t), where n is a node of the graph
 * and @a t is the current time parameter. @a force must return a Point
 * representing the node's force at time @a t.
 *
 * @a constraint is called as @a constraint(g, @t), where g is the graph
 * and @a t is the current time parameter.
 */
template <typename G, typename F, typename C>
double symp_euler_step(G& g, double t, double dt, F& force, C& constraint) {
  for (GraphType::node_iterator it = g.node_begin(); it < g.node_end(); ++it) {
    Node n = *it;
    n.value().new_p_ = (n.position() + n.value().v_ * dt);
#ifdef COLORS
    n.value().visit_status_ = node_state::MISSED;
#endif
  }

  constraint(g, t);

  for (GraphType::node_iterator it = g.node_begin(); it < g.node_end(); ++it) {
    (*it).set_position((*it).value().new_p_);
  }


  for (GraphType::node_iterator it = g.node_begin(); it < g.node_end(); ++it) {
    Node n = *it;
    node_state & state = n.value();
    state.v_ += force(n, t) * dt / state.m_;
  }

  return t + dt;
}

#ifdef COLORS

struct ConstraintColorFunctor {

  Color operator()(const Node & node) {
    return Color::make_heat(1 - ((int) node.value().visit_status_) / 3.0);
  }
};
#endif

void load_graph(GraphType & graph, const char * nodes_filename, const char * tets_filename) {
  // Read all Points and add them to the Graph
  std::ifstream nodes_file(nodes_filename);
  Point p;
  while (CS207::getline_parsed(nodes_file, p))
    graph.add_node(p);

  // Read all mesh squares and add their edges to the Graph
  std::ifstream tets_file(tets_filename);
  Tetrahedron t; // Reuse this type
  while (CS207::getline_parsed(tets_file, t)) {
    if (t.n[0] < graph.size() && t.n[1] < graph.size()
      && t.n[2] < graph.size() && t.n[3] < graph.size()) {
      graph.add_edge(graph.node(t.n[0]), graph.node(t.n[1]));
      graph.add_edge(graph.node(t.n[0]), graph.node(t.n[2]));
      // Diagonal edges: include as of HW2 #2
      graph.add_edge(graph.node(t.n[0]), graph.node(t.n[3]));
      graph.add_edge(graph.node(t.n[1]), graph.node(t.n[2]));
      graph.add_edge(graph.node(t.n[1]), graph.node(t.n[3]));
      graph.add_edge(graph.node(t.n[2]), graph.node(t.n[3]));
    }
  }

  // Print out the stats
  std::cout << graph.num_nodes() << " " << graph.num_edges() << std::endl;
}

void init_graph(GraphType & graph) {
  for (GraphType::node_iterator it = graph.node_begin(); it < graph.node_end(); ++it) {
    Node n = *it;
    node_state & state = n.value();
    state.v_ = Point();
    state.m_ = 1.0 / graph.size();
#ifdef COLORS
    state.visit_status_ = node_state::MISSED;
#endif
  }

  for (GraphType::edge_iterator it = graph.edge_begin(); it < graph.edge_end(); ++it) {
    Edge e = *it;
    edge_state & state = e.value();
    state.k_ = spring_k;
    state.rest_l_ = e.length();
    state.calced_ = false;
  }
}

int main(int argc, char* argv[]) {
  // check arguments
  if (argc < 2) {
    std::cerr << "Usage: mass_spring NODES_FILE TETS_FILE\n";
    exit(1);
  }
  bool quiet = (argc > 3);

  double stop = 0;
  if (quiet) {
    stop = atof(argv[3]);
    std::cout << "quiet mode" << std::endl;
  }

  GraphType graph(WORLD);
  load_graph(graph, argv[1], argv[2]);
  init_graph(graph);

  // Launch the SDLViewer
  CS207::SDLViewer viewer;
  auto node_map = viewer.empty_node_map(graph);
  if (!quiet) {
    viewer.launch();

    //display graph
    viewer.add_nodes(graph.node_begin(), graph.node_end(), node_map);
    viewer.add_edges(graph.edge_begin(), graph.edge_end(), node_map);
    viewer.center_view();
  }

  // Begin the mass-spring simulation
  const double dt = 0.0001;
  const double t_start = 0;
  const double t_end = 100;
  const double big_dt = dt * 100;

  GravityForce gravity;
  MassSpringForce springs;
  DampingForce damping(1.0 / graph.size());


  PlaneConstraint plane1 = PlaneConstraint(Point(0, 0, -0.75), Point(0, 0, 1));
  SphereConstraint sphere(Point(0.5, 0.5, -0.5), 0.15);
  SphereConstraint sphere1(Point(0.75, 0.25, -0.5), 0.15);
  SphereConstraint sphere2(Point(0.75, 0.75, -0.5), 0.15);
  SphereConstraint sphere3(Point(0.25, 0.25, -0.5), 0.15);
  SphereConstraint sphere4(Point(0.25, 0.75, -0.5), 0.15);
  ConstantConstraint<Node > node1(Point(0.0, 0.0, 0.0));
  ConstantConstraint<Node > node2(Point(1.0, 0.0, 0.0));

  auto force = make_combined_force(gravity, springs, damping);
  auto constraint = make_combined_constraint(node1, node2, plane1, sphere, sphere1, sphere2, sphere3, sphere4);

  CS207::Clock clock;

  double real_time = clock.elapsed();
  const GraphType::PerfStats & g_stats = graph.stats();
  for (double t = t_start; t < t_end; t += dt) {
    symp_euler_step(graph, t, dt, force, constraint);
    // Update viewer with nodes' new positions
    if (!quiet) {
#ifdef COLORS
      ConstraintColorFunctor colorer;
      viewer.add_nodes(graph.node_begin(), graph.node_end(), colorer, node_map);
#else
      viewer.add_nodes(graph.node_begin(), graph.node_end(), node_map);
#endif
    }
    double real_updated = clock.elapsed();
    if (fmod(t, big_dt) < dt && !quiet) {
      std::stringstream ss;
      double hit_ratio = (g_stats.neigh_total_ - g_stats.neigh_skipped_) / (double) g_stats.neigh_total_;
      graph.reset_stats();
      ss << std::fixed << std::setprecision(4) << hit_ratio << "    " << (big_dt * 100) / (real_updated - real_time);
      real_time = real_updated;
      viewer.set_label(ss.str());
    }
    if (quiet && real_updated > stop) {
      std::cout << "Virtual time elapsed: " << t << std::endl;
      break;
    }
  }

  return 0;
}
