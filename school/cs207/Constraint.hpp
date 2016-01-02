#ifndef CS207_CONSTRAINT_HPP
#define CS207_CONSTRAINT_HPP
#include <tuple>
#include <cmath>
#include "Point.hpp"
#include "Graph.hpp"
#include "BoundingBox.hpp"
#include "Collision.hpp"
#include "mass_spring_types.hpp"
#include "Collision.hpp"

#define USE_BB

/** @file Constraint.hpp
 * @brief Constraints and constraint functions. */


struct PlaneConstraint {
  Point p_, n_;
  BoundingBox b_;

  PlaneConstraint(const Point & p, const Point & n) : p_(p), n_(n) {
    assert(n_.l0_norm() == 1); //only along x, y, or z axis
    n_.normalize();
    Point diag(1.0, 1.0, 1.0);
    if (n_ < Point()) { //lex. comparison works with only one non-zero component
      diag *= -1;
    }
    diag -= n_;
    diag *= 100;
    b_ = BoundingBox(p_ + diag, p_ - diag);
    b_ |= p_ - (n_);
    b_ &= WORLD;
  }

  /** Constrain points to be "above" the plane
   *  Here, "above" means in the positive direction of the normal
   */
  template <typename G > void operator()(G & g, double t) const {
    (void) t;
#ifdef USE_BB
    auto end = g.node_end(b_);
    for (auto it = g.node_begin(b_); it != end; ++it) {
      if (true) {
#else
    for (auto it = g.node_begin(); it != g.node_end(); ++it) {
      if (b_.contains((*it).value().new_pos_)) {
#endif
        Point & np = (*it).value().new_pos_;
        np = (p_ + rejection_normal(np - p_, n_));
        Point & v = (*it).value().v_;
        v = rejection_normal(v, n_);
#ifdef COLORS
        (*it).value().update_status(node_state::CONSTRAINED);
#endif
      }
    }
  }
};

struct SphereConstraint {
  Point c_;
  double r2_;
  BoundingBox b_;

  SphereConstraint(const Point & c, const double & r) : c_(c), r2_(r * r), b_(BoundingBox(c, r)) {
  }

  /** Constrain points to be "above" the plane
   *  Here, "above" means in the positive direction of the normal
   */
  template <typename G > void operator()(G & g, double t) const {
    (void) t;
#ifdef USE_BB
    auto end = g.node_end(b_);
    for (auto it = g.node_begin(b_); it != end; ++it) {
#else
    for (auto it = g.node_begin(); it != g.node_end(); ++it) {
#endif
      Point & np = (*it).value().new_pos_;
      Point n = np - c_;
      double len_sq = n.squared_length();
      if (len_sq < r2_) {
        np = (c_ + (n * std::sqrt(r2_ / len_sq)));
        Point & v = (*it).value().v_;
        v = rejection(v, n);
#ifdef COLORS
        (*it).value().update_status(node_state::CONSTRAINED);
      } else {
        (*it).value().visit_status_ = std::max((*it).value().visit_status_, node_state::BOXED);
#endif
      }
    }
  }
};

template <typename N>
struct ConstantConstraint {
  const Point p_;
  bool found_;
  BoundingBox b_;
  N n_;

  ConstantConstraint(const Point & p) : p_(p), found_(false), b_(p_) {
  }

  /** Constrains a node at a particular point
   */
  template <typename G > void operator()(G & g, double t) {

    if (t == 0.0) {
#ifdef USE_BB
      auto end = g.node_end(b_);
      for (auto it = g.node_begin(b_); it != end; ++it) {
#else
      for (auto it = g.node_begin(); it != g.node_end(); ++it) {
#endif
        if ((*it).position() == p_) {
          n_ = (*it);
          found_ = true;
        }
      }
    }

    if (found_) {
      n_.value().new_pos_ = p_;
      n_.value().v_ = Point();
#ifdef COLORS
      n_.value().update_status(node_state::CONSTRAINED);
#endif
    }
  }
};

template <typename M>
struct BoundingMeshConstraint {
  const M & m_;
  typedef typename M::Triangle Triangle;
  double edge_len_;

  BoundingMeshConstraint(const M & m, double edge_len) : m_(m), edge_len_(edge_len) {
  }
  
  void set_edge_length(double length) {
    edge_len_ = length;
  }

  /** Constrains a graph/mesh to be inside of another mesh
   * The normals of the triangles in the bounding mesh are assumed to
   * point outside the mesh.
   */
  template <typename G > void operator()(G & g, double t) {
		(void) t;
    typename G::node_iterator end = g.node_end();
    for (typename G::node_iterator it = g.node_begin(); it != end; ++it) {
      typename G::Node n = *it;
      Collision::Collision<Triangle> c = Collision::surface_collision(n, m_, edge_len_);
      if (c.has_next()) {
        Triangle t = c.next();
        Point norm = t.normal();
        Point tp = t.node(0).position();
        Point & np = n.value().new_pos_;
        np = (tp + rejection_normal(np - tp, norm)) - (norm * .01); 
        //np = (tp + rejection_normal(np - tp, norm)) - (norm * .2);
        Point & v = n.value().v_;
        v -= (rejection_normal(v, norm) * 2);
        
        //for this triangle (upon which the collision is directed), update the associated 
        //velocities on the three nodes as an approx. of the collision impact
        auto sphere_node0 = t.node(0);
        auto sphere_node1 = t.node(1);
        auto sphere_node2 = t.node(2);
        collision3DSphere(1,n,sphere_node0);
        collision3DSphere(1,n,sphere_node1);
        collision3DSphere(1,n,sphere_node2); 
      }
    }
  }
};


struct PointCollisionConstraint {
  double epsilon_;

  PointCollisionConstraint(double epsilon) : epsilon_(epsilon) {
  }


  /** Handles the collisions and the processing for gas molecules colliding
  *  @param[in] m The mesh which represents the cloud (gas)
  *  @param[in] t current time
  *  @param[in] epsilon The distance at which we consider two nodes to be colliding
  *  @post: The nodes of Mesh m have their velocities and new points changed to their post positions after a
  *  collision, if there was a collision
  */
  template <typename M > void operator()(M & m, double t) {
    (void) t;
    //iterate through all the nodes
    for(auto it = m.node_begin(); it != m.node_end(); ++it) {

      //create collision object to iterate through all collisions with this node
      auto col = Collision::point_collision(*it,m,epsilon_);
    
      // Get the colliding node
      if(col.has_next()) {
        auto n1 = *it;
        auto n2 = col.next();

        while(n1.index() > n2.index() && col.has_next())
          n2 = col.next();


        //only proceed further if n1 has another unique node with larger index to collide with
        if( n1.index() < n2.index() && (n1.position()-n2.position()).length() <= epsilon_ ) {
	  collision3D(1,n1,n2);
      }
    }

  }
  }
};


/** rotates a vector counter-clockwise 30 degrees given two points on the plane
 *  @param[in] v Vector to be rotated
 *  @param[in] p1 Point1 on the plane
 *  @param[in] p2 Point2 on the plane
 *  @param[in] theta Angle to rotate in radians
 *  @return Vector rotated
 */
Point rotate_vector(Point v, Point p1, Point p2, double theta) {

  //Axis of rotation by cross product
  Point axis = v.cross(p1 - p2);

  //src: inside.mine.edu/~gmurray/ArbitraryAxisRotation/
  return Point(
    (-axis.x * (-axis.x * v.x - axis.y * v.y - axis.z * v.z)) * (1 - cos(theta)) + v.x * cos(theta) + (-axis.z * v.y + axis.y * v.z) * sin(theta),

    (-axis.y * (-axis.x * v.x - axis.y * v.y - axis.z * v.z)) * (1 - cos(theta)) + v.y * cos(theta) + (axis.z * v.x - axis.x * v.z) * sin(theta),

    (-axis.z * (-axis.x * v.x - axis.y * v.y - axis.z * v.z)) * (1 - cos(theta)) + v.z * cos(theta) + (-axis.y * v.x + axis.x * v.y) * sin(theta)
    );

}

/** Collision Computer for a node that is part of triangle being impacted by another node
 *  NOTE: This follows closely the code at plasmaphysics.org.uk/programs/coll3d_cpp.htm
 * @param[in] R restitution coefficient (0 to 1 where 1 is perfectly elastic)
 * @param[in] n1 Node1
 * @param[in] n2 One node of the triangle involved in the collision
 * @pre n2 is a node of a triangle which is itself a part of a polygon
 * @post n2 has correctly updated value().new_pos_ and value().v_
 * @post old @a n1.value().v_ == new @a n1.value().v_
 * @post old @a n1.value().new_pos_ == new @a n1.value().new_pos_
 * @return 0: no error, 1: no collision, 2: error
 **/
template <typename NODE, typename TRINODE>
int collision3DSphere(double R, NODE & n1, TRINODE & n2) {


  double pi, r12, m21, d, v, theta2, phi2, st, ct, sp, cp, fvz1r,
    thetav, phiv, dr, alpha, beta, sbeta, cbeta, t, a, dvz2;
  Point p1 = n1.value().new_pos_;
  Point p2 = n2.value().new_pos_;
  double m1 = n1.value().mass_; 
  double m2 = n2.value().mass_;
  Point v1 = n1.value().v_; 
  Point v2 = n2.value().v_;
  //should be the radius, but I'll just make it half the distance between
  double r1 = 0.99 * 0.5 * (p1 - p2).length();
  double r2 = r1;


  //     **** initialize some variables ****
  pi = acos(-1.0E0);
  r12 = r1 + r2;
  m21 = m2 / m1;

  Point p21 = p2 - p1;

  Point v21 = v2 - v1;

  Point v_cm = Point(m1 * v1.x + m2 * v2.x, m1 * v1.y + m2 * v2.y, m1 * v1.z + m2 * v2.z);
  v_cm /= (m1 + m2);



  /**** calculate relative distance and relative speed ***/
  d = p21.length();
  v = v21.length();

  /**** return if distance between balls smaller than sum of radii ****/
  if (d < r12)
    return 2;

  /**** return if relative speed = 0 ****/
  if (v == 0)
    return 1;


  /**** shift coordinate system so that ball 1 is at the origin ***/
  p2 = p21;

  /**** boost coordinate system so that ball 2 is resting ***/
  v1 = v21;
  v1 *= -1;

  /**** find the polar coordinates of the location of ball 2 ***/
  theta2 = acos(p2.z / d);
  if (p2.x == 0 && p2.y == 0)
    phi2 = 0;
  else
    phi2 = atan2(p2.y, p2.x);
  st = sin(theta2);
  ct = cos(theta2);
  sp = sin(phi2);
  cp = cos(phi2);


  /**** express the velocity vector of ball 1 in a rotated coordinate **/
  //         system where ball 2 lies on the z-axis ******
  Point v1r = Point(ct * cp * v1.x + ct * sp * v1.y - st * v1.z, cp * v1.y - sp * v1.x, st * cp * v1.x + st * sp * v1.y + ct * v1.z);
  fvz1r = v1r.z / v;
  if (fvz1r > 1)
    fvz1r = 1; // fix for possible rounding errors
  else if (fvz1r<-1)
    fvz1r = -1;
  thetav = acos(fvz1r);
  if (v1r.x == 0 && v1r.y == 0)
    phiv = 0;
  else
    phiv = atan2(v1r.y, v1r.x);


  //**** calculate the normalized impact parameter ***
  dr = d * sin(thetav) / r12;


  //**** return old positions and velocities if balls do not collide ***
  if (thetav > pi / 2 || fabs(dr) > 1)
    return 1;

  //**** calculate impact angles if balls do collide ***
  alpha = asin(-dr);
  beta = phiv;
  sbeta = sin(beta);
  cbeta = cos(beta);


  //**** calculate time to collision ***
  t = (d * cos(thetav) - r12 * sqrt(1 - dr * dr)) / v;


  //**** update positions and reverse the coordinate shift ***
  p2 = p2 + v2 * t + p1;
  p1 = (v1 + v2) * t + p1;

  //  ***  update velocities ***

  a = tan(thetav + alpha);
  dvz2 = 2 * (v1r.z + a * (cbeta * v1r.x + sbeta * v1r.y)) / ((1 + a * a)*(1 + m21));
  Point v2r = Point(a * cbeta*dvz2, a * sbeta*dvz2, dvz2);
  v1r = v1r - v2r*m21;

  //**** rotate the velocity vectors back and add the initial velocity
  //           vector of ball 2 to retrieve the original coordinate system ****
  v1 = v2 + Point(ct * cp * v1r.x - sp * v1r.y + st * cp * v1r.z, ct * sp * v1r.x + cp * v1r.y + st * sp * v1r.z, ct * v1r.z - st * v1r.x);
  v2 = v2 + Point(ct * cp * v2r.x - sp * v2r.y + st * cp * v2r.z, ct * sp * v2r.x + cp * v2r.y + st * sp * v2r.z, ct * v2r.z - st * v2r.x);

  // ***  velocity correction for inelastic collisions ***

  v1 = v_cm + (v1 - v_cm) * R;
  v2 = v_cm + (v2 - v_cm) * R;

  n2.value().v_ = v2 / 3;
  n2.value().new_pos_ = p2;

  return 0;
}


/** Collision Computer, NOTE src: plasmaphysics.org.uk/programs/coll3d_cpp.htm
 * @param[in] R restitution coefficient (0 to 1 where 1 is perfectly elastic)
 * @param[in] n1 Node1
 * @param[in] n2 Node2
 * @post n1 and n2 have correctly updated new positons
 * @return 0: no error, 1: no collision, 2: error
 **/
template <typename NODE>
int collision3D(double R, NODE & n1, NODE & n2) {


  double pi, r12, m21, d, v, theta2, phi2, st, ct, sp, cp, fvz1r,
    thetav, phiv, dr, alpha, beta, sbeta, cbeta, t, a, dvz2;

  Point p1 = n1.value().new_pos_;
  Point p2 = n2.value().new_pos_;
  double m1 = n1.value().mass_;
  double m2 = n2.value().mass_;
  Point v1 = n1.value().v_;
  Point v2 = n2.value().v_;
  //should be the radius, but I'll just make it half the distance between
  double r1 = 0.999*0.5 * (p1 - p2).length();
  double r2 = r1;


  //     **** initialize some variables ****
  pi = acos(-1.0E0);
  r12 = r1 + r2;
  m21 = m2 / m1;

  Point p21 = p2 - p1;

  Point v21 = v2 - v1;

  Point v_cm = Point(m1 * v1.x + m2 * v2.x, m1 * v1.y + m2 * v2.y, m1 * v1.z + m2 * v2.z);
  v_cm /= (m1 + m2);



  /**** calculate relative distance and relative speed ***/
  d = p21.length();
  v = v21.length();

  /**** return if distance between balls smaller than sum of radii ****/
  if (d < r12)
    return 2;

  /**** return if relative speed = 0 ****/
  if (v == 0)
    return 1;


  /**** shift coordinate system so that ball 1 is at the origin ***/
  p2 = p21;

  /**** boost coordinate system so that ball 2 is resting ***/
  v1 = v21;
  v1 *= -1;

  /**** find the polar coordinates of the location of ball 2 ***/
  theta2 = acos(p2.z / d);
  if (p2.x == 0 && p2.y == 0)
    phi2 = 0;
  else
    phi2 = atan2(p2.y, p2.x);
  st = sin(theta2);
  ct = cos(theta2);
  sp = sin(phi2);
  cp = cos(phi2);


  /**** express the velocity vector of ball 1 in a rotated coordinate **/
  //         system where ball 2 lies on the z-axis ******
  Point v1r = Point(ct * cp * v1.x + ct * sp * v1.y - st * v1.z, cp * v1.y - sp * v1.x, st * cp * v1.x + st * sp * v1.y + ct * v1.z);
  fvz1r = v1r.z / v;
  if (fvz1r > 1)
    fvz1r = 1; // fix for possible rounding errors
  else if (fvz1r<-1)
    fvz1r = -1;
  thetav = acos(fvz1r);
  if (v1r.x == 0 && v1r.y == 0)
    phiv = 0;
  else
    phiv = atan2(v1r.y, v1r.x);


  //**** calculate the normalized impact parameter ***
  dr = d * sin(thetav) / r12;


  //**** return old positions and velocities if balls do not collide ***
  if (thetav > pi / 2 || fabs(dr) > 1)
    return 1;

  //**** calculate impact angles if balls do collide ***
  alpha = asin(-dr);
  beta = phiv;
  sbeta = sin(beta);
  cbeta = cos(beta);


  //**** calculate time to collision ***
  t = (d * cos(thetav) - r12 * sqrt(1 - dr * dr)) / v;


  //**** update positions and reverse the coordinate shift ***
  p2 = p2 + v2 * t + p1;
  p1 = (v1 + v2) * t + p1;

  //  ***  update velocities ***

  a = tan(thetav + alpha);
  dvz2 = 2 * (v1r.z + a * (cbeta * v1r.x + sbeta * v1r.y)) / ((1 + a * a)*(1 + m21));
  Point v2r = Point(a * cbeta*dvz2, a * sbeta*dvz2, dvz2);
  v1r = v1r - v2r*m21;

  //**** rotate the velocity vectors back and add the initial velocity
  //           vector of ball 2 to retrieve the original coordinate system ****
  v1 = v2 + Point(ct * cp * v1r.x - sp * v1r.y + st * cp * v1r.z, ct * sp * v1r.x + cp * v1r.y + st * sp * v1r.z, ct * v1r.z - st * v1r.x);
  v2 = v2 + Point(ct * cp * v2r.x - sp * v2r.y + st * cp * v2r.z, ct * sp * v2r.x + cp * v2r.y + st * sp * v2r.z, ct * v2r.z - st * v2r.x);

  // ***  velocity correction for inelastic collisions ***

  v1 = v_cm + (v1 - v_cm) * R;
  v2 = v_cm + (v2 - v_cm) * R;

  n1.value().v_ = v1;
  n1.value().new_pos_ = p1;
  n2.value().v_ = v2;
  n2.value().new_pos_ = p2;


  return 0;
}

template <typename...CL>
struct CombinedConstraint {
  typedef std::tuple < CL&...> constraints_type;
  constraints_type constraints_;

  CombinedConstraint(CL&... constraints) : constraints_(constraints...) {
  }

  /** Returns the sum of two forces applying to @a n at time @a t.
   *
   */
  template <typename G > void operator()(G & g, double t) {
    //    return add(constraints_, n, t);
    for_eacher<G, std::tuple_size<constraints_type>::value > ::call(constraints_, g, t);
  }

  template<typename G, size_t I>
    struct for_eacher {

    static void call(constraints_type & constraints, G & g, double t) {
      std::get < I - 1 > (constraints) (g, t);
      for_eacher <G, I - 1 > ::call(constraints, g, t);
    }
  };

  template<typename G>
    struct for_eacher<G, 0 > {

    static void call(constraints_type & constraints, G & g, double t) {
      (void) constraints;
      (void) g;
      (void) t;
      return;
    }
  };
};

/** Combines a set of constraint functors into a single constraint functor
 * @param ... constraints a set of constraint functors
 * @return a constraint functor that applies all @a constraints on it's arguments
 */
template<typename... C>
CombinedConstraint<C...> make_combined_constraint(C&... constraints) {
  return CombinedConstraint < C...>(constraints...);
}


#endif
