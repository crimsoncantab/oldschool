#ifndef CS207_FORCE_HPP
#define CS207_FORCE_HPP
#include <tuple>
#include "Point.hpp"
#include "Graph.hpp"

/** @file Force.hpp
 * @brief Helpful forces and force functions. */

static constexpr double grav = 9.80665;

/** Structure to add the force of gas to a triangle face */
struct GasTForce {
private:
  double C_;

public:

  /** Default constructor */
  GasTForce() : C_(0) {
  }

  GasTForce(double C) : C_(C) {
  }

  /** Return the new force on triangle @a tri at time @a t with volume @a v */
  template <typename T >
    Point operator()(T tri, double t, double v) {
    (void) t;
    return tri.normal() * ((C_ / v) * (tri.area()));
  }
};

/** The force on each node is the sum of forces on all incident faces */
struct IncidentFaceForce {

  /** Return the sum of all forces applied on all faces incident to
   * node @a n at time @a t 
   * @pre Triangle values must support a force value (Point) */
  template <typename N >
    Point operator()(N n, double t) {
    (void) t;
    Point force = Point();
    for (auto it = n.triangle_begin(); it != n.triangle_end(); ++it)
      force += (*it).value().force_;
    return force;
  }
};

struct DampingForce {
  const double damp_c_;

  DampingForce(const double damp_c) : damp_c_(damp_c) {
  }

  /** Return the damping force applying to @a n at time @a t.
   *
   */
  template <typename N > Point operator()(N n, double t) const {
    (void) t;

    return n.value().v_ * -damp_c_;
  }
};

struct MassSpringForce {

  /** Return the mass-spring force applying to @a n at time @a t.
   *
   */
  template <typename N > Point operator()(N n, double t) const {
    (void) t;

    Point spring_f;
    for (auto it = n.edge_begin(); it < n.edge_end(); ++it) {
      auto e = *it;
      auto & e_state = e.value();
      if (e_state.calced_) {

        spring_f -= e_state.force_;
      } else {
        Point dir = e.node2().position() - n.position();
        dir.normalize();
        dir *= e_state.k_ * (e.length() - e_state.rest_l_);
        spring_f += dir;
        e_state.force_ = dir;
      }
      e_state.calced_ = !e_state.calced_; //for other node
      //    (void) t;
      //    Point force = Point(0,0,0);
      //    for (auto it = n.edge_begin(); it != n.edge_end(); ++it) {
      //      force -= (*it).unit_edge() * (*it).value().K *
      //        ((*it).length() - (*it).value().L);
      //    }
      //    return force;

    }
    return spring_f;
  }
};

struct GravityForce {
  const Point grav_;

  GravityForce(const double accel = grav) : grav_(0.0, 0.0, -accel) {
  }

  /** Return the gravitational force applying to @a n at time @a t.
   *
   */
  template <typename N > Point operator()(N n, double t) const {
    (void) t;

    //return grav_ * n.value().m_;
    return grav_ * n.value().m_;
  }
};


class DampingForceBasic {
  double c_;
 public:
  DampingForceBasic(double c)
    : c_(c) {
  }
  /** Damping Force on Node @a n at time @a t*/
  template <typename N>
  Point operator()(N n, double t) {
    (void) t;
    Point total_damping_force = n.value().v_*-1*c_;

    return total_damping_force;  
  }
};

/** Dashpot spring force for 3D objects
 *
 * Implements a dashpot spring force
 */
class DashpotSpringForce {
  double c_;
 public:
  /** DashpotSpringForce constructor
   * @param[in] c A damping constant
   */
  DashpotSpringForce(double c)
    : c_(c) {
  }
  /**Dashpot Spring Force on Node @a n at time @a t*/
  template <typename N>
  Point operator()(N n, double t) {
    (void) t;
    Point total_dashpot_spring_force = Point(0, 0, 0);
    for (auto edgei = n.edge_begin(); edgei != n.edge_end(); ++edgei) {
      auto edge = *edgei;

      Point edge_diff = edge.node1().position() - edge.node2().position();
                        Point norm = edge_diff / edge.length();
                        Point edge_velocity_diff = edge.node1().value().v_ - edge.node2().value().v_;
                        //(c*(v_i - v_j) dot (x_i - x_j)) / lengthOf(x_i - x_j):
                        double intermediate_value = ( edge_velocity_diff.dot(edge_diff) /  edge.length() ) * c_;
                        double intermediate_value2 = ( (edge.length() - edge.value().L) * (edge.value().K) ) + intermediate_value;
      total_dashpot_spring_force += (norm) * (intermediate_value2) * -1; 
    }

    return total_dashpot_spring_force;  
  }
};

/** Structure for the Force of air */
struct AirForce {
private:
  double c_;
  Point w_;

public:

  /** Default constructor */
  AirForce() : c_(0), w_(Point()) {
  }

  AirForce(double c, Point w) : c_(c), w_(w) {
  }

  /** Return the force of air on node @a n at time @a t */
  template <typename N >
    Point operator()(N n, double t) {
    (void) t;
    Point normal = n.normal();
    normal.normalize();
    return normal * (c_ * (w_ - n.value().v_).dot(normal));
  }
};

template <typename...FL>
struct CombinedForce {
  typedef std::tuple < FL&...> forces_type;
  forces_type forces_;

  CombinedForce(FL&... forces) : forces_(forces...) {
  }

  /** Returns the sum of all combined forces applying to @a n at time @a t.
   *
   */
  template <typename N > Point operator()(N & n, double t) {
    return for_eacher<N, std::tuple_size<forces_type>::value > ::add(forces_, n, t);
  }

private:

  template<typename N, size_t I>
    struct for_eacher {

    static Point add(forces_type & forces, N & n, double t) {
      return std::get < I - 1 > (forces) (n, t) + for_eacher <N, I - 1 > ::add(forces, n, t);
    }
  };

  template<typename N>
    struct for_eacher<N, 0 > {

    static Point add(forces_type & forces, N & n, double t) {
      (void) forces;
      (void) n;
      (void) t;
      return Point();
    }
  };
};

/** Combines a set of force functors into a single force functor
 * @param ... forces a set of force functors
 * @return a force functor that evaluates all @a forces on it's arguments
 * and sums their results.
 */
template<typename... F>
CombinedForce<F...> make_combined_force(F&... forces) {
  return CombinedForce < F...>(forces...);
}

#endif
