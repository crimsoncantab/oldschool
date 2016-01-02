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
#include <math.h>
#include "Simplex.hpp"
#include "Mesh.hpp"
#include "Collision.hpp"

#include "Constraint.hpp"
#include "Force.hpp"

static constexpr double EPSILON = 0.05;
static constexpr double SIMULATION_POINTS = 1000;
static constexpr double BOUNDING_BOX_DELTA = 0.20;
static constexpr double K_VALUE = 10;
double TEMP_MULTI_FACTOR = 1.0;


struct CloudColor {

  template <typename NODE >
    Color operator()(const NODE & n) {
    (void)n;
    return Color(1,0,0);
  }
};

struct SurfaceColor {

  template <typename NODE >
    Color operator()(const NODE & n) {
    (void)n;
    return Color(0,.5,.5);
  }
};

struct node_value {
  Point v_;
  double mass_;
  Point new_pos_;
};

struct edge_value {
  double K;  //spring constant
  double L;  //spring rest length
};

struct triangle_value {
  double mass_;
};


typedef Mesh<node_value,edge_value,triangle_value> MeshType;
typedef SpatialGraph<node_value, int> GraphType;

class VelocityAction : public Action {
  const GraphType & cloud_;
  const double mult_;

public:

  VelocityAction(const GraphType & cloud, double mult) : cloud_(cloud), mult_(mult) {
  }

  virtual void act(typename std::vector<unsigned>::iterator first,
    typename std::vector<unsigned>::iterator last) {
    for (; first != last; ++first) {
      GraphType::Node n = cloud_.node(*first);
      n.value().v_ *= mult_;
    }
  }
};

/** Change a mesh's nodes according to a step of the symplectic Euler
 *    method with the given node force and constraint.
 * @param[in,out] mesh Mesh on which to apply the step
 * @param[in] t the current time (useful for time-dependent forces)
 * @param[in] dt the time step
 * @param[in] force function object defining the force per node
 * @param[in] constraint constraint object defining constraints on nodes
 *
 * @return the next time step (usually @a t + @a dt)
 *
 * @a force is called as @a force(n, @a t), where n is a node of the tetrahedral mesh
 * and @a t is the current time parameter. @a force must return a Point
 * representing the node's force at time @a t.
 */
template <typename M, typename F, typename C>
double symp_euler_step(M& mesh, double t, double dt, F& force, C& constraint) {
  for (auto ni = mesh.node_begin(); ni != mesh.node_end(); ++ni) {
    auto node = *ni;
    //update position:
    Point v_i = node.value().v_;
    v_i *= dt;
    node.value().new_pos_ = node.position() + v_i;
  }

  //apply constraint(s)
  constraint(mesh, t);
  
  for (auto it = mesh.node_begin(); it < mesh.node_end(); ++it) {
    (*it).set_position((*it).value().new_pos_);
  }

  for (auto ni = mesh.node_begin(); ni != mesh.node_end(); ++ni) {
    auto node = *ni;
    //update v_i:
    Point force_point = force(node, t);
    force_point *= (dt / node.value().mass_);
    node.value().v_ = node.value().v_ + force_point;
  }

  return t + dt;
}

/** @class IdealGasProcess
 * Approximates the movement of gas molecules in a spherical object
 */
template <typename F>
struct IdealGasProcess {
  MeshType & sphere_;
  GraphType & cloud_;
  int sphere_id_;
  int cloud_id_;
  F force_;
  VelocityAction up_;
  VelocityAction down_;

  /** Public constructor for IdealGasProcess
   * @param[in,out] s Mesh that represents the sphereical container
   * @param[in,out] c Mesh that represents the inner molecules
   * @param[in] node_force Force object (mass-spring-esque) applied to s
   */
  IdealGasProcess(MeshType & s, GraphType & c, F node_force) : sphere_(s), cloud_(c),
  sphere_id_(-1), cloud_id_(-1), force_(node_force), up_(c, 2), down_(c, .5) {
  }
  template <typename VIEW>
    void init(VIEW * view) {
    view->add_nodes(sphere_.node_begin(), sphere_.node_end(), sphere_id_, DefaultNormal(), CloudColor(), NormalVector());
    view->add_edges(sphere_.edge_begin(), sphere_.edge_end(), sphere_id_);
    view->add_triangles(sphere_.triangle_begin(), sphere_.triangle_end(), sphere_id_);

    view->add_nodes(cloud_.node_begin(), cloud_.node_end(), cloud_id_, NoNormal());

    view->center_view();

    up_.key = SDLK_UP;
    down_.key = SDLK_DOWN;
    view->add_action(&up_, cloud_id_);
    view->add_action(&down_, cloud_id_);
  }

  template <typename VIEW>
    void operator()(double t, double dt, VIEW * view) {

    //calculate longest edge length on each iteration (used to construct the bounding box in the BoundingMeshConstraint)
    double longest_edge_length = 0.0;
    for(auto it = sphere_.edge_begin(); it != sphere_.edge_end(); ++it) {
      MeshType::Edge edge = *it;
      longest_edge_length = (edge.length() > longest_edge_length) ? 
        edge.length() : longest_edge_length;
    }

    BoundingMeshConstraint<MeshType> sphere_constraint_instance = BoundingMeshConstraint<MeshType>(sphere_, longest_edge_length);
    PointCollisionConstraint pcc = PointCollisionConstraint(EPSILON);
    auto constraint = make_combined_constraint(sphere_constraint_instance, pcc);

    auto nc = make_combined_constraint();
    auto nf = make_combined_force();

    symp_euler_step(sphere_, t, dt, force_, nc);  //a force is applied to the sphere
    symp_euler_step(cloud_, t, dt, nf, constraint); //the collision constraints are applied to the cloud nodes

    view->add_nodes(sphere_.node_begin(), sphere_.node_end(), sphere_id_, DefaultNormal(), SurfaceColor(), NormalVector());
    view->add_nodes(cloud_.node_begin(), cloud_.node_end(), cloud_id_, NoNormal(), CloudColor());
    view->set_label(t);   
  }
};

template <typename M, typename M2, typename NF>
IdealGasProcess<NF> make_process(M & sphere, M2 & cloud, NF & node_force) {
  return IdealGasProcess<NF> (sphere, cloud, node_force);
}

/** Add nodes randomly within a spherical boundary
 * @param[in,out] mesh Mesh in which to add the points
 * @param[in] center Approx. center of spherical boundary 
 * @param[in] radius Approx. radius of spherical boundary
 * @post old @a mesh.num_nodes() == new @a mesh.num_nodes() + SIMULATION_POINTS
 * @post old @a mesh.num_edges() == new @a mesh.num_edges()
 * @post old @a mesh.num_triangles() == new @a mesh.num_triangles()
 *
 * This function is designed to add nodes with positions that are within an approximately
 * spherical boundary that is in reality a polygon.
 */
void populate_graph(GraphType &mesh, Point center, double radius) {
  // Read all Points and add them to the Mesh
  double m = 1;
  for (int i = 0; i < SIMULATION_POINTS; i++) {
		radius *= 0.99; // ensure that no points are initialized outside the polygon, which isn't a perfect sphere
    Point uniformSpherePoint = Point(radius*CS207::random(-1,1),
        radius*CS207::random(-1,1),radius*CS207::random(-1,1));

    while( (uniformSpherePoint).length() > radius) {
      uniformSpherePoint = Point(radius*CS207::random(-1,1),
        radius*CS207::random(-1,1),radius*CS207::random(-1,1));
    }

    Point p = center + uniformSpherePoint;
    Point v(CS207::random(-1,1),CS207::random(-1,1),CS207::random(-1,1));
    node_value val = {v, m, Point(0,0,0)};
    mesh.add_node(p, val);
  }
  
  // Print out the stats
  std::cout << mesh.num_nodes() << " "
    << mesh.num_edges() << std::endl;
}

/** Load a mesh object from a file
 * @param[in,out] mesh Mesh in which to store the file data; supports nodes and triangles
 * @param[in] nodes char * of a text file of nodes
 * @param[in] triangles char * of a text file of triangles
 * @post old @a mesh.num_nodes() <= new @a mesh.num_nodes()
 * @post old @a mesh.num_edges() <= new @a mesh.num_edges()
 * @post old @a mesh.num_triangles() <= new @a mesh.num_triangles()
 */
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

/** Set initial conditions of the sphere
 * @param[in,out] mesh Mesh of which to set initial conditions
 * @pre mesh has node.value() struct with Point v_, double mass_, and Point new_pos_ members
 * @pre mesh has edge.value() struct with double K and double L members
 * @post old @a mesh.num_nodes() == new @a mesh.num_nodes()
 * @post old @a mesh.num_edges() == new @a mesh.num_edges()
 */
template <typename M>
void set_sphere_initial_conditions(M& mesh) {
  //Set initial conditions for nodes:
  for (auto ni = mesh.node_begin(); ni != mesh.node_end(); ++ni) {
    auto node = *ni;
    node_value node_data;
    node_data.v_ = Point(0,0,0);  //zero initial velocity
    node_data.mass_ = (double)1.0 / mesh.num_nodes(); 
		node_data.new_pos_ = node.position();
    node.value() = node_data;
  }

  //Set initial conditions for edges: 
  for (auto edgei = mesh.edge_begin(); edgei != mesh.edge_end(); ++edgei) {
    auto one_edge = *edgei;
    edge_value edge_data;
    edge_data.K = K_VALUE;  
    edge_data.L = one_edge.length();  
    one_edge.value() = edge_data;
  }
}

int main(int argc, char* argv[]) {
  // check arguments
  if (argc < 2) {
    std::cerr << "Usage: mass_spring NODES_FILE TETS_FILE\n";
    exit(1);
  }
  
  MeshType sphere;
  load_mesh(sphere, argv[1], argv[2]);

  //determine approximate center
  Point center;
  for(auto it=sphere.node_begin(); it != sphere.node_end(); ++it) {
    center = center + (*it).position();
  }

  //assuming a relatively uniform sphere to calculate radius
  double radius = (center - sphere.node(0).position()).length();

  //set initial triangle areas as the mass of the sphere triangles
  for(auto it = sphere.triangle_begin(); it != sphere.triangle_end(); ++it) {
    MeshType::Triangle tri = *it;
    tri.value().mass_ = tri.area();
  }

  //set additional initial conditions on the sphere:
  set_sphere_initial_conditions(sphere);

  GraphType cloud;
  populate_graph(cloud, center, radius);
  double dt = 0.001;

  DashpotSpringForce dashpot_spring_force((double)1.0 / sphere.num_nodes());
  DampingForce damping_force(0.01);
  auto node_force = make_combined_force(dashpot_spring_force, damping_force);

  auto process = make_process(sphere, cloud, node_force);
  Simulator viewer;
  viewer.launch();

  double t_start = 0;
  double t_end = 10;

  // Begin the simulation
  viewer.run(process, t_start, t_end, dt);

  return 0;
}
#endif
