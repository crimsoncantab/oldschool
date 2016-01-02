#ifndef SHALLOW_WATER_CPP
#define SHALLOW_WATER_CPP

/**
 * @file shallow_water.cpp
 * Implementation of a shallow water system using Mesh
 *
 * @brief Reads in two files specified on the command line.
 * First file: 3D point list (one per line) defined by three doubles
 * Second file: Triangles (one per line) defined by 3 indices into the point list
 */

#include "Simulator.hpp"
#include "CS207/Util.hpp"
#include "Point.hpp"
#include "Color.hpp"
#include <fstream>
#include <deque>

#include "Simplex.hpp"
#include "Mesh.hpp"

// Standard gravity (average gravity at Earth's surface) in meters/sec^2
static constexpr double grav = 9.80665;

/** Water column characteristics */
struct QVar {
  double h; ///< Height of fluid
  double hu; ///< Height times average x velocity of column
  double hv; ///< Height times average y velocity of column

  /** Default constructor.
   *
   * A default water column is 1 unit high with no velocity. */
  QVar()
    : h(1), hu(0), hv(0) {
  }

  /** Construct the given water column. */
  QVar(double h_, double hu_, double hv_)
    : h(h_), hu(hu_), hv(hv_) {
  }

  /** Add @a q to this qvar. */
  QVar& operator+=(const QVar & q) {
    h += q.h;
    hu += q.hu;
    hv += q.hv;
    return *this;
  }

  /** Scale this qvar by a factor. */
  QVar& operator*=(double d) {
    h *= d;
    hu *= d;
    hv *= d;
    return *this;
  }
};

/** Function object for calculating shallow-water flux.
 *          |e
 *   T_k    |---> n = (nx,ny)   T_m
 *   qk     |                   qm
 *          |
 * @param[in] nx, ny Defines the 2D outward normal vector n = (@a nx, @a ny)
 *            from triangle T_k to triangle T_m. The length of n is equal to the
 *            the length of the edge, |n| = |e|.
 * @param[in] dt The time step taken by the simulation. Used to compute the
 *               Lax-Wendroff dissipation term.
 * @param[in] qk The values of the conserved variables on the left of the edge.
 * @param[in] qm The values of the conserved variables on the right of the edge.
 * @return The flux of the conserved values across the edge e
 */
struct EdgeFluxCalculator {

  QVar operator()(double nx, double ny, double dt, const QVar& qk, const QVar & qm) {
    double e_length = sqrt(nx * nx + ny * ny);
    nx /= e_length;
    ny /= e_length;

    // The velocities normal to the edge
    double wk = (qk.hu * nx + qk.hv * ny) / qk.h;
    double wm = (qm.hu * nx + qm.hv * ny) / qm.h;

    // Lax-Wendroff local dissipation coefficient
    double vk = sqrt(grav * qk.h) + sqrt(qk.hu * qk.hu + qk.hv * qk.hv) / qk.h;
    double vm = sqrt(grav * qm.h) + sqrt(qm.hu * qm.hu + qm.hv * qm.hv) / qm.h;
    double a = dt * std::max(vk*vk, vm * vm);

    // Helper values
    double scale = 0.5 * e_length;
    double gh2 = 0.5 * grav * (qk.h * qk.h + qm.h * qm.h);

    // Simple flux with dissipation for stability
    return QVar(scale * (wk * qk.h + wm * qm.h) - a * (qk.h - qm.h),
      scale * (wk * qk.hu + wm * qm.hu + gh2 * nx) - a * (qk.hu - qm.hu),
      scale * (wk * qk.hv + wm * qm.hv + gh2 * ny) - a * (qk.hv - qm.hv));
  }
};

struct NodeColor {

  template <typename NODE >
    Color operator()(const NODE & n) {
    return Color::make_heat(1.0 - std::sqrt(n.value().hu * n.value().hu + n.value().hv * n.value().hv));
  }
};

struct NodeVector {

  template <typename NODE >
    Point operator()(const NODE & n) {
    (void) n;
    return Point(n.value().hu, n.value().hv, 0) / 10;
  }
};

// HW3B: Placeholder for Mesh Type!
// Define NodeData, EdgeData, TriData, etc

struct triangle_value {
  QVar qvar_;
  double area_;
};

struct edge_value {
  Point out_normal_;
  unsigned out_triangle_;
};

typedef Mesh<QVar, edge_value, triangle_value> MeshType;

class Bumper : public Action<MeshType> {

  virtual void act(typename std::vector<typename MeshType::node_type>::iterator first,
    typename std::vector<typename MeshType::node_type>::iterator last) {
    for (; first != last; ++first) {
      MeshType::Node n = *first;
      for (MeshType::incident_triangle_iterator iti = n.triangle_begin(); iti != n.triangle_end(); ++iti) {
        (*iti).value().qvar_.h += .1;
      }
    }
  }
};

/** Integrate a hyperbolic conservation law defined over the mesh m
 * with flux functor f by dt in time.
 */
template <typename MESH, typename FLUX>
double hyperbolic_step(MESH& m, FLUX& f, double t, double dt) {

  std::vector<struct QVar> newValues;

  //calculate the flux outgoing from each of the triangles
  for (MeshType::triangle_iterator it = m.triangle_begin(); it != m.triangle_end(); ++it) {

    QVar sumOfFlux(0, 0, 0);
    MeshType::Triangle t = *it;

    //iterate over each of the adjacent triangles and calculate Flux along the shared edge
    for (MeshType::triangle_neighbor_iterator it2 = t.triangle_begin(); it2 != t.triangle_end(); ++it2) {
      MeshType::Triangle adj = *it2;

      //find the edge on the boundary of t and adj
      edge_value & ev = it2.edge().value();

      //calculate unit normal along the edge
      Point normal = ev.out_normal_;
      if (ev.out_triangle_ != t.index()) {
        normal *= -1;
      }

      //add to the total flux the flux leaving this border
      sumOfFlux += f(normal.x, normal.y, dt, t.value().qvar_, adj.value().qvar_);
    }

    //iterate over ghost edges if any exist
    for (int i = 0; i < 3; i++) {

      //if only 1 triangle, must be a ghost edge
      if (t.edge(i).num_triangles() == 1) {
        edge_value & ev = t.edge(i).value();
        Point normal = ev.out_normal_;
        if (ev.out_triangle_ != t.index()) {
          normal *= -1;
        }
        struct QVar ghost = QVar(t.value().qvar_.h, 0, 0);
        sumOfFlux += f(normal.x, normal.y, dt, t.value().qvar_, ghost);
      }
    }

    //weight the flux leaving by the area
    sumOfFlux *= dt / t.value().area_;

    //add the totalFlux leaving to a vector to update currentFlux later
    newValues.push_back(sumOfFlux);
  }

  //iterate through stored leaving fluxes and calculate new current flux
  int i;
  MeshType::triangle_iterator it3;
  for (it3 = m.triangle_begin(), i = 0; it3 != m.triangle_end(); ++it3, ++i) {
    MeshType::Triangle t = *it3;
    t.value().qvar_ += newValues[i];
  }

  return t + dt;
}

/** Convert the triangle-averaged values to node-averaged values for viewing. */
template <typename MESH>
void post_process(MESH& m) {

  //iterate through all the nodes
  for (MeshType::node_iterator it = m.node_begin(); it != m.node_end(); ++it) {
    MeshType::Node n = *it;
    double total_area = 0;
    QVar total_qvar(0, 0, 0);

    //calculate the new height as the average of all the triangle heights incident to the node with weight associated to the area
    for (MeshType::incident_triangle_iterator it2 = n.triangle_begin(); it2 != n.triangle_end(); ++it2) {
      MeshType::Triangle t = *it2;
      total_area += t.value().area_;
      total_qvar.h += t.value().area_ * t.value().qvar_.h;
      total_qvar.hu += t.value().area_ * t.value().qvar_.hu;
      total_qvar.hv += t.value().area_ * t.value().qvar_.hv;
    }
    n.value().h = total_qvar.h / total_area;
    n.value().hu = total_qvar.hu / total_area;
    n.value().hv = total_qvar.hv / total_area;
    Point p = n.position();
    p.z = n.value().h;
    n.set_position(p);
  }

}

/** Calculate the minimum edge length in a graph
 *@pre: @a m has at least one edge
 */
template <typename MESH>
double minimum_edge_length(MESH& m) {
  if (m.num_edges() == 0) {
    return 0;
  }

  MeshType::edge_iterator it = m.edge_begin();
  double min_len = (*it).length();
  ++it;

  //iterate over all the edges and look for the minimum edge length
  for (; it != m.edge_end(); ++it) {
    min_len = std::min<double>(min_len, (*it).length());
  }
  return min_len;
}

/** Calculate the maximum height of any triangle in the graph
 *@pre: @a m has at least one triangle
 */
template <typename MESH>
double max_height(MESH& m) {
  if (m.num_triangles() == 0)
    return 0;

  MeshType::triangle_iterator it = m.triangle_begin();
  double max_h = (*it).value().qvar_.h;
  ++it;

  //iterate over all the triangles and look for the maximum height
  for (; it != m.triangle_end(); ++it) {
    max_h = std::max<double>(max_h, (*it).value().qvar_.h);
  }
  return max_h;

}

struct ShallowWaterProcess {
  MeshType & m;

  ShallowWaterProcess(MeshType & tmp) : m(tmp) {

  }

  template <typename VIEW>
    void init(VIEW * view) {
    view->add_nodes(m.node_begin(), m.node_end(), DefaultNormal(),
      NodeColor(), NodeVector());
    view->add_edges(m.edge_begin(), m.edge_end());
    view->add_triangles(m.triangle_begin(), m.triangle_end());
    view->center_view();
  }

  template <typename VIEW>
    void operator()(double t, double dt, VIEW * view) {
    EdgeFluxCalculator f;

    //calculate the new current fluxes
    hyperbolic_step(m, f, t, dt);

    // Update viewer with nodes' new positions
    post_process(m);

    view->add_nodes(m.node_begin(), m.node_end(), DefaultNormal(),
      NodeColor(), NodeVector());
    view->set_label(t);

  }
};
typedef Simulator<MeshType, ShallowWaterProcess> SimulatorType;

class Deleter : public Action<MeshType> {
  MeshType & m_;
  SimulatorType & s_;

  virtual void act(typename std::vector<typename MeshType::node_type>::iterator first,
    typename std::vector<typename MeshType::node_type>::iterator last) {
    for (; first != last; ++first) {
      MeshType::Node n = *first;
      m_.remove_node(n);
    }
    s_.clear();
    s_.add_nodes(m_.node_begin(), m_.node_end(), DefaultNormal(),
      NodeColor(), NodeVector());
    s_.add_edges(m_.edge_begin(), m_.edge_end());
    s_.add_triangles(m_.triangle_begin(), m_.triangle_end());
  }

public:

  Deleter(MeshType & m, SimulatorType & s) : m_(m), s_(s) {

  }
};

double init_mesh(MeshType & mesh, int setting) {
  //Set the initial settings of the simulation based on the third argument
  //given 1: we simulate the large column of water
  //given 2: Dam break and waterfall
  //otherwise: pebble drop
  for (MeshType::edge_iterator eit = mesh.edge_begin(); eit != mesh.edge_end(); ++eit) {
    MeshType::Edge e = *eit;
    e.value().out_triangle_ = 0;
  }
  for (MeshType::triangle_iterator it = mesh.triangle_begin(); it != mesh.triangle_end(); ++it) {
    MeshType::Triangle t = *it;
    Point avg(0, 0, 0);
    for (int i = 0; i < 3; ++i) {
      avg += t.node(i).position();
      MeshType::Edge e = t.edge(i);
      edge_value & ev = e.value();
      if (ev.out_triangle_ == 0) {
        ev.out_normal_ = t.normal(i) * e.length();
        ev.out_triangle_ = t.index();
      }
    }
    avg /= 3;

    t.value().area_ = t.area();
    switch (setting) {
      case 1:
        t.value().qvar_ = QVar(((pow(avg.x - 0.75, 2) + pow(avg.y, 2) - 0.15 * 0.15) > 0) ? 1.75 : 1, 0, 0);
        break;
      case 2:
        t.value().qvar_ = QVar((avg.x > 0) ? 1.75 : 1, 0, 0);
        break;
      default: //0
        t.value().qvar_ = QVar(1 - 0.75 * exp(-80 * (pow(avg.x - 0.75, 2) + pow(avg.y, 2))), 0, 0);
        break;
    }
  }

  post_process(mesh);

  // CFL stability condition requires dt <= dx / max|velocity|
  // For the shallow water equations with u = v = 0 initial conditions
  //   we can compute the minimum edge length and maximum original water height
  //   to set the time-step
  return 0.25 * minimum_edge_length(mesh) / (sqrt(grav * max_height(mesh)));
}

void load_mesh(MeshType & mesh, char * nodes, char * triangles) {
  // Read all Points and add them to the Mesh
  std::ifstream nodes_file(nodes);
  Point p;
  while (CS207::getline_parsed(nodes_file, p)) {
    mesh.add_node(p);
  }
  // Read all mesh triangles and add their edges to the Mesh
  std::ifstream tris_file(triangles);
  Triangle simplex;
  while (CS207::getline_parsed(tris_file, simplex)) {
    mesh.add_triangle(mesh.node(simplex.n[0]), mesh.node(simplex.n[1]), mesh.node(simplex.n[2]));
  }
  // Print out the stats
  std::cout << mesh.num_nodes() << " "
    << mesh.num_edges() << " "
    << mesh.num_triangles() << std::endl;

}

int main(int argc, char* argv[]) {
  // Check arguments
  if (argc < 3) {
    std::cerr << "Usage: shallow_water NODES_FILE TETS_FILE [INITIAL_COND]\n";
    exit(1);
  }


  MeshType mesh;
  load_mesh(mesh, argv[1], argv[2]);
  // Perform any needed precomputation
  // Compute the minimum edge length and maximum water height for computing dt
  // Set the initial conditions
  double dt = init_mesh(mesh, (argc < 4) ? 0 : atoi(argv[3]));

  // Launch the SDLViewer
  //CS207::SDLViewer viewer;
  Echoer<MeshType> e;
  e.key = SDLK_e;
  Bumper b;
  b.key = SDLK_b;
  Simulator<MeshType, ShallowWaterProcess> viewer =
    Simulator<MeshType, ShallowWaterProcess > (ShallowWaterProcess(mesh));
//  Deleter d(mesh, viewer);
//  d.key = SDLK_r;
  viewer.add_action(&e);
  viewer.add_action(&b);
//  viewer.add_action(&d);
  viewer.launch();

  double t_start = 0;
  double t_end = 10;

  // Begin the simulation
  viewer.run(t_start, t_end, dt);

  return 0;
}
#endif
