#ifndef COLLISION_HPP
#define COLLISION_HPP

#include <unordered_set>
#include <vector>
#include "assert.h"

//#include "BoundingBox.hpp"
#include "Point.hpp"


/** @file Collision.hpp
 * @brief Provide functionality for detecting collisions between graph-like
 * objects. Two interfaces are provided, a basic interface, and a collision
 * event handling system that allows registration of objects and collision
 * handler callbacks.
 */


namespace Collision {


  /** Specifies the type of intersection between two objects.
   *
   * intersect_yes and intersect_no represent intersections with 100% confidence.
   * intersect_degenerate represents a situation in which we cannot determine if
   * an intersection exists or not. The conditions under which each of these will
   * be returned is specified in each method's specification.
   *
   * Would be defined as a forward reference, but doesn't seem possible...
   */
  enum intersect_t {
    intersect_yes,
    intersect_no,
    intersect_degenerate
  };


  /*********************
   * Forward references
   *********************/

  /* Intersect primitive prototypes */
  struct Plane;
  struct tri_type;
  template <typename T> static intersect_t triangle_segment_intersect(T t, Point p1, Point p2);
  template <typename T> static Point surface_norm(T t);
  template <typename T> static Plane containing_plane(T t);
  static intersect_t plane_segment_intersect(Plane plane, Point q, Point r);
  static int volume_sign(tri_type t, Point p);
  static int volume_sign(Point t1, Point t2, Point t3, Point p);

  /** @class Collision::Collision
   * @brief A java-style iterator to iterate over objects colliding with a given node. A
   * forward iterator.
   */
  template <typename O>
  class Collision {

  public:
    /** The type of object the node is colliding with. */
    typedef O collision_object_type;
    
    /** A default constructor. */
    Collision(){
      collision_objects_index_ = 0;
    }
    
    /** Adds an object to the list of objects colliding with the given node. */
    void add_object(collision_object_type o){
      collision_objects_.push_back(o);
    }

    /** Returns true if there are more items to iterator over, false otherwise.*/
    bool has_next(){
      if(collision_objects_.empty() || 
         collision_objects_index_ == collision_objects_.size()){
        return false;
      }
      return true;
    }

    /** Returns the next object in the list of those colliding with the given node.
     * @pre has_next() == true
     */
    collision_object_type next(){
      assert(has_next());
      ++collision_objects_index_;
      return collision_objects_[collision_objects_index_ - 1];
    }

  private:
    //the list of objects colliding with the node
    std::vector<O> collision_objects_;
    //the position in the list of collision objects
    unsigned collision_objects_index_;
  };


  /** Returns true if the ray defined by (@a center, @a point) is itersecting the triangle
   * @a t or false otherwise. An intersection includes those points on the edge of the
   * triangle. This will also return true if the ray is in the same plane as @a t.
   *
   * @param t The triangle to check.
   * @param p The second point of the ray.
   * @param center The origin point of the ray.
   * @return True if there is an intersection between the triangle and the ray, false 
   * otherwise.
   */
  template <typename TRIANGLE>
  bool point_in_triangle(TRIANGLE t, Point p, Point center) {
    intersect_t coll = triangle_segment_intersect(t, p, center);
    //return true;    
    if(coll == intersect_no)
      return false;
    else
      return true;
  }

  /** Returns a Collision object containing the triangle if a collision was found or
   * and empty Collision object if no collision was found.
   *
   * @param n The cloud node to use for collision detection
   * @param m The sphere mesh containing the triangles to check.
   * @edge_length The maximum edge length of the triangle in the sphere mesh.
   * @return A Collision iterator object containing a collision triangle, if any.
   */
  template <typename NODE, typename MESH>
  static Collision<typename MESH::Triangle> surface_collision(NODE n, MESH & m, double edge_length){
    Collision<typename MESH::Triangle> collision_tri;

    //BoundingBox bb = BoundingBox(n.position(), edge_length);      
    BoundingBox bb = BoundingBox(n.value().new_pos_, edge_length); 
    //iterate through sphere nodes in bounding box to check for collision
    for(auto it2 = m.node_begin(bb); it2 != m.node_end(bb); ++it2) {
        //find triangle involved in collision -- assume only one exists, if any
        typename MESH::Node s_node = *it2;
        for(auto tri_it = s_node.triangle_begin(); tri_it != s_node.triangle_end(); ++tri_it) {
          typename MESH::Triangle closest_tri = *tri_it;
          //if we have a collision of the cloud node with the triangle
          //draw the ray from the center point, to ensure a triangle is grabbed for large deltas and sphere deformation
          if(point_in_triangle(closest_tri, n.value().new_pos_, Point(0,0,0))) {
          //if(point_in_triangle(closest_tri, n.value().new_pos_, n.position())) {
            collision_tri.add_object(closest_tri);
            //std::cout << "collide" << std::endl;
            return collision_tri;
          }       
        }
      }
    return collision_tri;
  }
  
  /** Returns a Collision object containing any cloud nodes found colliding with @a n or
   * and empty Collision object if no collisions were found.
   *
   * @param n The cloud node to use for collision detection
   * @param m The cloud mesh containing the nodes to check.
   * @param epsilon The distance that two nodes must be within each other to declare
   * a collision.
   * @return A Collision iterator object containing all collision nodes, if any.
   */
  template <typename NODE, typename MESH>
  static Collision<typename MESH::Node> point_collision(NODE n, MESH & m, double epsilon){
    Collision<typename MESH::Node> collision_list;

    //create a bounding box using the epsilon as the size
    BoundingBox point_sphere = BoundingBox(n.value().new_pos_, epsilon);

    //iterate through all cloud nodes within the bounding box
    for(auto ni = m.node_begin(point_sphere); ni != m.node_end(point_sphere); ++ni){
      typename MESH::Node n2 = *ni;

      /**double check that the nodes don't equal eachother and that they are within
         an epsilon distance of each other. */
      if((n.value().new_pos_ - n2.value().new_pos_).length() < epsilon && n != n2){
        collision_list.add_object(n2);
      }
    }

    return collision_list;
  }


/****************************************************
 * Low-level collision detection "private" interface
 ****************************************************/

/** Check if a triangle @a t is intersected by a line segment (@a q,@a r).
 * @param t A triangle in 3D space
 * @param q The "start" endpoint of the line segment (endpoint of a ray)
 * @param r The "end" endpoint of the line segment (direction of a ray)
 * @return intersect_yes if the segment interior crosses exactly one point on the triangle interior
 *         intersect_no if the segment does not cross the triangle, or if @a q lies on the triangle
 *         intersect_degenerate if the segment crosses only the border of the triangle (node or edge),
 *           or if it crosses more than one point on the triangle's interior (that is, the segment lies
 *           in the same plane as the triangle).
 */
template <typename T>
static intersect_t triangle_segment_intersect(T t, Point q, Point r) {
  intersect_t plane_seg = plane_segment_intersect(containing_plane(t), q, r);

  if (plane_seg == intersect_no || plane_seg == intersect_degenerate)
    return plane_seg;
  else
    return triangle_segment_cross(t, q, r);
}

/** A plane with equation Ax+By+Cz=D */
struct Plane {
  /** Surface normal of the plane, in either direction. Coefficients A,B,C are fields x,y,z */
  Point norm_;
  /** D coefficient of plane. */
  double d_;

  Plane(Point norm, double d)
  : norm_(norm), d_(d) {
  }
};

/** Write a Plane to an output stream, in the form "Ax+By+Cz=D" */
inline std::ostream& operator<<(std::ostream& s, const Plane& p) {
  return (s << p.norm_.x << "x + " << p.norm_.y << "y + " << p.norm_.z << "z = " << p.d_);
}

/** Return a unit length vector representing a normal to the plane containing triangle @a t.
 * The normal's direction (inward or outward) is undefined.
 */
template <typename T>
static Point surface_norm(T t) {
  Point u = t.node(1).position() - t.node(0).position();
  Point v = t.node(2).position() - t.node(0).position();
  return u.cross(v);
}

/** Return the plane in which triangle @a t lies. */
template <typename T>
static Plane containing_plane(T t) {
  Point norm = surface_norm(t);
  // Ax+By+Cz=D  =>  (x,y,z) \cdot (A,B,C) = D
  double d = t.node(0).position().dot(norm);
  return Plane(norm,d);
}

/** Return the intersection type between a plane @a plane and a line segment (@a q,@a r).
 * @param plane A plane with which to check intersection
 * @param q The "start" endpoint of the line segment (endpoint of a ray)
 * @param r The "end" endpoint of the line segment (direction of a ray)
 * @return intersect_yes if the segment interior crosses the plane at exactly one point
 *         intersect_no if no part of the segment touches the plane
 *         intersect_degenerate if the segment lies within the plane or touches the plane at an endpoint
 */
static intersect_t plane_segment_intersect(Plane plane, Point q, Point r) {
  // parametric equation of segment: p(t) = q + t*(r-q)
  // substituted into equation of a plane, p(t) \cdot (A,B,C) = D
  //   => t = (D - q \cdot N) / ((r-q) \cdot N)

  double numerator = plane.d_ - q.dot(plane.norm_);
  double denominator = (r-q).dot(plane.norm_);

  // TODO think about rounding errors for code below, print debug statement for now
  if (fabs(denominator) < 0.000000001)
    //std::cerr << "Almost parallel. plane:" << plane << " q-r:(" << q-r << ") dot:" << denominator << std::endl;

  // segment parallel to the plane iff segment \cdot norm == 0
  if (denominator == 0.0) {
    if (numerator == 0.0) // segment on plane
      return intersect_degenerate;
    else // segment not on plane
      return intersect_no;
  }

  // p(t) is the point where the line defined by the segment would intersect the plane
  double t = numerator / denominator;

  // p(t) is strictly on the interior of the segment if t in (0,1)
  if (0.0 < t && t < 1.0)
    return intersect_yes;

  // p(t) is exactly at endpoint q if t==0 (check fractions to avoid floating-point math)
  // we define this situation to not be an intersection
  if (numerator == 0.0)
    return intersect_no;

  // p(t) is exactly at endpoint r if t==1 (check fractions to avoid floating point math)
  // degenerate since we don't know if this should count as a crossing with the ray
  if (numerator == denominator) {
    // BUT, this case should never happen, if we are choosing points on a bounding box + epsilon
    // OTOH, with floating-point math the if-statement will almost never be true, so we should be
    // using some threshold, which must be smaller than bb_threshold, or we can pick a ray that
    // will correctly reach this point.
    //assert(false);
    return intersect_degenerate;
  }

  // p(t) is not on the segment if t not in [0,1]
  // we already checked all values in the range, so just return no
  return intersect_no;
}

/** Check if a segment crosses a triangle, given endpoints on opposite sides of the triangle's plane.
 * @pre @a q and @a r are on opposite sides of containing_plane(@a t)
 * @pre neither @a q or @a r lie on the triangle
 * @param t The triangle to check for intersection with
 * @param q The "first" endpoint of the segment (the endpoint of the ray it represents)
 * @param r The "second" endpoing of the segment (the direction of the ray it represents)
 * @return intersect_yes if the segment crosses at a point strictly interior to the triangle
 *         intersect_no if the segment does not cross the triangle
 *         intersect_degenerate if the intersection is on the boundary of the triangle
 */
template <typename T>
static intersect_t triangle_segment_cross(T t, Point q, Point r) {
  // compute volume sign of each tetrahedra formed by an edge and the segment
  // we choose the nodes in the edges in a consistent order, so they will be CCW or CW
  // (which orientation doesn't matter - we check for both in all cases)
  int vol0 = volume_sign(q, t.node(0).position(), t.node(1).position(), r);
  int vol1 = volume_sign(q, t.node(1).position(), t.node(2).position(), r);
  int vol2 = volume_sign(q, t.node(2).position(), t.node(0).position(), r);

  // intersect triangle interior if all same (non-zero) sign
  if ((vol0 > 0 && vol1 > 0 && vol2 > 0) ||
      (vol0 < 0 && vol1 < 0 && vol2 < 0))
    return intersect_yes;

  // no intersection if any have opposite sign
  if ((vol0 > 0 || vol1 > 0 || vol2 > 0) &&
      (vol0 < 0 || vol1 < 0 || vol2 < 0))
    return intersect_no;

  // degenerate triangle or segment co-planar with triangle if all are zero
  // TODO should we assert if triangle is denegerate? No, it probably has been previously handled
  if (vol0 == 0 && vol1 == 0 && vol2 == 0)
    return intersect_degenerate;

  // segment intersects vertex if two zeros
  if ((vol0 == 0 && vol1 == 0) ||
      (vol1 == 0 && vol2 == 0) ||
      (vol2 == 0 && vol0 == 0))
    return intersect_degenerate;

  // segment intersects edge if one zero
  if (vol0 == 0 || vol1 == 0 || vol2 == 0)
    return intersect_degenerate;

  // one of these cases always must be true
  // if we ever get here, we missed some case, or volume_sign is incorrect
  assert(false);
  return intersect_degenerate;
}

/** Convenience triangle type with 3 points. */
struct tri_type {
  static constexpr size_t num_points = 3;
  Point points[num_points];

  Point operator[](size_t i) const {
    assert(i < num_points);
    return points[i];
  }
  Point& operator[](size_t i) {
    assert(i < num_points);
    return points[i];
  }
};

/** Return the sign of the volume of the tetrahedron formed by triangle @a t and point @p.
 * @param t The base of the tetrahedron
 * @param p The point connected to the base of the tetrahedron
 * @return +1 if the points of the base follow the right-hand rule w.r.t. the outward normal
 *         -1 if the points of the base follow the right-hand rule w.r.t. the inward normal
 *          0 if the points do not form a non-degenerate tetrahedron (ie, all are co-planar)
 */
static int volume_sign(tri_type t, Point p) {
  // the signed volume is 1/6 the determinant of
  //   | t[0].x t[0].y t[0].z 1 |
  //   | t[1].x t[1].y t[1].z 1 |
  //   | t[2].x t[2].y t[2].z 1 |
  //   | t[3].x t[3].y t[3].z 1 |

  Point a = t.points[0] - p;
  Point b = t.points[1] - p;
  Point c = t.points[2] - p;
  Point d = Point(0,0,0);

  double vol1 =
    - (a.z-d.z)*(b.y-d.y)*(c.x-d.x)
    + (a.y-d.y)*(b.z-d.z)*(c.x-d.x)
    + (a.z-d.z)*(b.x-d.x)*(c.y-d.y)
    - (a.x-d.x)*(b.z-d.z)*(c.y-d.y)
    - (a.y-d.y)*(b.x-d.x)*(c.z-d.z)
    + (a.x-d.x)*(b.y-d.y)*(c.z-d.z);

  if (vol1 > 0) return 1;
  if (vol1 < 0) return -1;
  return 0;
}
static int volume_sign(Point t1, Point t2, Point t3, Point p) {
  tri_type tri;
  tri[0] = t1;
  tri[1] = t2;
  tri[2] = t3;
  return volume_sign(tri, p);
}



} // end namespace Collision
#endif
