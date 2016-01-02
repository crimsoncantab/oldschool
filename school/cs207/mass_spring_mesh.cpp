/**
 * @file mass_spring_mesh_closed.cpp
 * Implementation of mass-spring system using Mesh on closed surfaces.
 *
 * @brief Reads in two files specified on the command line.
 * First file: 3D Points (one per line) defined by three doubles
 * Second file: Tetrahedra (one per line) defined by 4 indices into the point
 * list
 */

#include "CS207/Util.hpp"
#include "Mesh.hpp"
#include "Point.hpp"
#include "Tetrahedron.hpp"
#include "Color.hpp"
#include "BoundingBox.hpp"
#include <fstream>
#include "Simulator.hpp"
#include "Constraint.hpp"
#include "Force.hpp"
#include "Simplex.hpp"
#include "Mesh.hpp"
#include "mass_spring_types.hpp"

// Define your Mesh type; this is a placeholder.
typedef Mesh<node_state, edge_state, triangle_state> MeshType;
typedef MeshType::Node Node;
typedef MeshType::Edge Edge;

/** Computes the volume of the mesh @a g
 * @pre g must be a closed surface */
template <typename G>
double volume(G& g) {
  double v = 0;
  for (auto it = g.triangle_begin(); it != g.triangle_end(); ++it) {
    auto t = *it;
    v += (t.normal().z / 2) * (t.node(0).position().z + t.node(1).position().z
      + t.node(2).position().z) / 3;
  }
  return v;
}

/** Change a mesh's nodes according to a step of the symplectic Euler
 *    method with the given node force.
 * @param[in,out] g mesh
 * @param[in] t the current time (useful for time-dependent forces)
 * @param[in] dt the time step
 * @param[in] force function object defining the force per node
 * @param[in tforce trianglular force function object defining forces on
 *   triangular faces
 * @pre G::node_value_type supports node_value
 * @pre G::triangle_value_type supports force (Point)
 * @post G::triangle_value_type force is updated to reflect the force on that
 *   triangular face
 * @return the next time step (usually @a t + @a dt)
 *
 * @a force is called as @a force(n, @a t), where n is a node of the mesh
 * and @a t is the current time parameter. @a force must return a Point
 * representing the node's force at time @a t.
 *
 * the new force on each node must reflect the sum of all forces on that node,
 * in addition to any forces acting on incident trianglular faces, located in
 * the force component of G::triangle_value_type
 */
//template <typename F, typename TF>
//double symp_euler_step(MeshType& g, double t, double dt, F force, TF tforce) {
//  // iterate to change all node positions
//  for (auto it = g.node_begin(); it != g.node_end(); ++it) {
//    Node n = *it;
//    n.set_position(n.position() + n.value().v_ * dt);
//  }
//  for (auto it = g.node_begin(); it != g.node_end(); ++it) {
//    Node n = *it;
//    n.value().v_ += force(n, t) * dt / n.value().m_;
//  }
//
//  return t + dt;
//}
template <typename F, typename TF, typename C>
double symp_euler_step(MeshType& g, double t, double dt, F& force, TF tforce, C& constraint) {
  for (MeshType::node_iterator it = g.node_begin(); it < g.node_end(); ++it) {
    Node n = *it;
    n.value().new_p_ = (n.position() + n.value().v_ * dt);
#ifdef COLORS
    n.value().visit_status_ = node_state::MISSED;
#endif
  }

  constraint(g, t);

  for (MeshType::node_iterator it = g.node_begin(); it < g.node_end(); ++it) {
    (*it).set_position((*it).value().new_p_);
  }

  // update the forces on all triangles
  double v = volume(g);
  for (MeshType::triangle_iterator it = g.triangle_begin(); it != g.triangle_end(); ++it) {
    auto tri = *it;
    tri.value().force_ = tforce(tri, t, v);
  }
  
  // iterate to change all node forces and velocity
  for (MeshType::node_iterator it = g.node_begin(); it < g.node_end(); ++it) {
    Node n = *it;
    node_state & state = n.value();
    state.v_ += force(n, t) * dt / state.m_;
  }

  return t + dt;
}

/** Node position function object for use in the SDLViewer. */
struct NodePosition {

  template <typename NODE >
    Point operator()(const NODE & n) {
    return n.position();
  }
};

struct NodeColor {

  template <typename NODE >
    Color operator()(const NODE & n) {
//    return Color(1, 0, 0);
        return Color::make_heat(1.5 - n.value().v_.length());
  }
};

struct NodeVector {

  template <typename NODE >
    Point operator()(const NODE & n) {
    return n.normal() / 20;
  }
};

/*
class Bumper : public Action<MeshType> {
  virtual void act(typename std::vector<typename MeshType::node_type>::iterator first,
    typename std::vector<typename MeshType::node_type>::iterator last) {
    for (; first != last; ++first) {
      MeshType::Node n = *first;
      for (MeshType::incident_triangle_iterator iti = n.triangle_begin(); iti != n.triangle_end();  ++iti) {
        (*iti).value().qvar_.h += .1;
      }
    }
  }
};
 */
template <typename F, typename T>
struct MassSpringProcess {
  MeshType & m;
  //  double mass_;
  //  double v_init_;
  PlaneConstraint constraint_;
  F force_;
  T triangle_force_;

  MassSpringProcess(MeshType & tmp, F node_force, T tri_force) : m(tmp),
    constraint_(Point(0, 0, -1), Point(0, 0, 1)), force_(node_force),
    triangle_force_(tri_force) {

  }

  template <typename VIEW>
    void init(VIEW * view) {
    view->add_nodes(m.node_begin(), m.node_end(),
      DefaultColor(), NodePosition());
    view->add_edges(m.edge_begin(), m.edge_end());
    view->add_triangles(m.triangle_begin(), m.triangle_end());
    view->center_view();
  }

  template <typename VIEW>
    void operator()(double t, double dt, VIEW * view) {
    symp_euler_step(m, t, dt, force_, triangle_force_, constraint_);

//    constraint_(m, t);
    //    PlaneZConstraint(-1)(m,t);

    // Update viewer with nodes' new positions
    view->add_nodes(m.node_begin(), m.node_end(),
      NodeColor(), NodePosition(), NodeVector());
    view->set_label(t);
  }
};

template <typename M, typename NF, typename TF>
MassSpringProcess<NF, TF> make_process(M & mesh, NF & node_force, TF & triangle_force) {
  return MassSpringProcess<NF, TF > (mesh, node_force, triangle_force);
}

template < typename P>
Simulator<MeshType, P> make_simulator(P & process) {
  return Simulator<MeshType, P > (process);
}

int main(int argc, char* argv[]) {
  // check arguments
  if (argc < 2) {
    std::cerr << "Usage: mass_spring NODES_FILE TETS_FILE\n";
    exit(1);
  }

  MeshType mesh;

  // Read all Points and add them to the Mesh
  std::ifstream nodes_file(argv[1]);
  Point p;
  while (CS207::getline_parsed(nodes_file, p))
    mesh.add_node(p);

  // Read all mesh squares and add their edges to the Mesh
  std::ifstream tris_file(argv[2]);
  Triangle t;
  while (CS207::getline_parsed(tris_file, t)) {
    // HW3B: Need to implement add_triangle this before this can be used!
    mesh.add_triangle(mesh.node(t.n[0]), mesh.node(t.n[1]), mesh.node(t.n[2]));
  }

  // Print out the stats
  std::cout << mesh.num_nodes() << " "
    << mesh.num_edges() << " "
    << mesh.num_triangles() << std::endl;

  // Begin the mass-spring simulation
  double dt = 0.0001;
  double t_start = 0;
  double t_end = 10;
  double spring_constant = 100.0;
  double mass = 1.0 / mesh.num_nodes();
  double damping = 1.0 / mesh.num_nodes();
  Point v_init = Point();
  Point wind = Point(0, 0, -1);
  double wind_constant = 2;
  double gas_constant = 10;

  // initialize node values
  for (auto it = mesh.node_begin(); it != mesh.node_end(); ++it) {
    node_state nv = {mass, v_init, Point()};
    (*it).value() = nv;
  }

  // initialize edge values
  for (auto it = mesh.edge_begin(); it != mesh.edge_end(); ++it) {
    edge_state ev = {spring_constant, (*it).length(), false, Point()};
    (*it).value() = ev;
  }

  Echoer<MeshType> e;
  e.key = SDLK_e;
  GravityForce gravity_force;
  MassSpringForce mass_spring_force;
  DampingForce damping_force(damping);
  AirForce wind_force(wind_constant, wind);
  IncidentFaceForce face_force;
  auto node_force = make_combined_force(gravity_force, mass_spring_force,
    damping_force, wind_force, face_force);
  auto tri_force = GasTForce(gas_constant); //Link, save the princess!

  auto process = make_process(mesh, node_force, tri_force);
  auto viewer = make_simulator(process);
  viewer.add_action(&e);
  viewer.launch();
  viewer.run(t_start, t_end, dt);
  return 0;
}

