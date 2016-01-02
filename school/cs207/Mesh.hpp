#ifndef CS207_MESH_HPP
#define CS207_MESH_HPP

/**
 * @file Mesh.hpp
 */

#include <vector>
#include "Graph.hpp"
#include "SpatialGraph.hpp"
#include <assert.h>

// Automatically derive !=, <=, >, and >= from a class's == and <
using namespace std::rel_ops;

/** @class Mesh
 * @brief A template for 3D undirected meshs.
 *
 * Users can add and retrieve nodes and edges. Edges are unique (there is at
 * most one edge between any pair of nodes). */
template <typename N, typename E, typename T >
class Mesh {
private:
  struct triangle_info;
  typedef struct triangle_info dual_node_value_type;
  struct dual_edge_info;
  typedef dual_edge_info dual_edge_value_type;
  typedef Graph<triangle_info, dual_edge_value_type> dual_graph_type;
  typedef std::vector<typename dual_graph_type::Node> dual_node_list_type;

  typedef struct {
    N value_;
    dual_node_list_type triangles_;
  } primal_node_value_type;

  typedef struct {
    E value_;
    dual_node_list_type triangles_;
  } primal_edge_value_type;
  typedef SpatialGraph<primal_node_value_type, primal_edge_value_type> primal_graph_type;

  struct triangle_info {
    T value_;
    typename primal_graph_type::Edge edges_[3];
    typename primal_graph_type::Node nodes_[3];
  };

  struct dual_edge_info {
    //    bool padding_;
    typename primal_graph_type::Edge primal_edge_;
  };

public:

  /** Type of this mesh. */
  typedef Mesh<N, E, T> mesh_type;

  /** Type of mesh nodes. */
  class Node;
  /** Synonym for Node (following STL conventions). */
  typedef Node node_type;

  /** Type of mesh edges. */
  class Edge;
  /** Synonym for Edge (following STL conventions). */
  typedef Edge edge_type;

  /** Type of mesh triangles. */
  class Triangle;
  /** Synonym for Triangle (following STL conventions). */
  typedef Triangle triangle_type;

  /** Type of node, triangle, and edge values. */
  typedef N node_value_type;
  typedef E edge_value_type;
  typedef T triangle_value_type;


  /** Type of indexes and sizes. */
  typedef unsigned size_type;

  /* base type for most iterators */
  template <typename IT, typename V>
  class base_iterator;

  /** Type of node iterators. */
  typedef base_iterator<typename primal_graph_type::node_iterator, Node> node_iterator;

  /** Type of edge iterators, which iterate over all mesh edges. */
  typedef base_iterator<typename primal_graph_type::edge_iterator, Edge> edge_iterator;

  /** Type of triangle iterators, which iterate over all mesh triangles. */
  typedef base_iterator<typename dual_graph_type::node_iterator, Triangle> triangle_iterator;

  /** Iterator class for nodes within a certain bounding box. */
  typedef base_iterator<typename primal_graph_type::neighborhood_iterator, Node> neighborhood_iterator;

    /** Type of incident iterators, which iterate over the edges that touch
        a given node. */
    typedef base_iterator<typename primal_graph_type::incident_iterator, Edge> incident_iterator;

  /** Type of incident iterators, which iterate over the triangles that touch
      a given node. */
  typedef base_iterator<typename dual_node_list_type::const_iterator, Triangle> incident_triangle_iterator;

  /** Type of neighbor iterators, which iterate over the triangle/edge pairs
      that touch a given triangle. */
  class triangle_neighbor_iterator;


  // CONSTRUCTOR AND DESTRUCTOR

  /** Construct an empty mesh. */
  Mesh() : primal_(), dual_() {
  }

  /** Destroy the mesh. */
  ~Mesh() {
  }


  /* NODES */

  /** @class Mesh::Node
   * @brief Class representing the mesh's nodes.
   *
   * Node objects are used to access information about the Mesh's nodes.
   * The mesh offers several ways to access nodes, including iterators
   * and the node() function.
   */
  class Node {
  public:

    /** Construct an invalid node.
     *
     */
    Node() : m_(NULL), primal_node_() {
    }

    /** Return the mesh that contains this node. */
    mesh_type * mesh() const {
      return m_;
    }

    /** Return this node's position. */
    Point position() const {
      return primal_node_.position();
    }

    /** Set this node's position.
     */
    void set_position(const Point & p) {
      primal_node_.set_position(p);
    }

    /** Return the surface normal on this point. O(degree()) */
    Point normal() const {
      Point n;
      for (incident_triangle_iterator it = triangle_begin(); it != triangle_end(); ++it)
        n += (*it).normal();
      return n.normalize();
    }

    /** Return this node's value. */
    const node_value_type & value() const {
      return primal_node_.value().value_;
    }

    node_value_type & value() {
      return primal_node_.value().value_;
    }

    /** Return this node's index, a number in the range [0, mesh_size). */
    size_type index() const {
      return primal_node_.index();
    }

    /** Test whether this node and @a x are equal.
     *
     * Equal nodes have the same mesh and the same index. */
    bool operator==(const Node& x) const {
      return m_ == x.m_ && primal_node_ == x.primal_node_;
    }

    /** Test whether this node is less than @a x in the global order.
     *
     */
    bool operator<(const Node& x) const {
      return (m_ < x.m_) || (m_ == x.m_ && primal_node_ < x.primal_node_);
    }

    /** Return the number of edges incident to this node.
     *
     * Complexity: O(1) */
    size_t degree() const {
      return primal_node_.degree();
    }

    /** Return the number of triangles incident to this node.
     *
     * Complexity: O(1) */
    size_t num_triangles() const {
      return primal_node_.value().triangles_.size();
    }

    /** Returns an iterator referring to the first edge incident to this node.
     *
     * Complexity: O(1) */
    incident_iterator edge_begin() const {
      return incident_iterator(m_, primal_node_.edge_begin());
    }

    /** Returns an iterator referring to the past-the-end edge incident to this node.
     *
     * Complexity: O(1) */
    incident_iterator edge_end() const {
      return incident_iterator(m_, primal_node_.edge_end());
    }

    /** Returns an iterator referring to the first triangle incident to this node.
     *
     * Complexity: O(1) */
    incident_triangle_iterator triangle_begin() const {
      return incident_triangle_iterator(m_, primal_node_.value().triangles_.begin());
    }

    /** Returns an iterator referring to the past-the-end triangle incident to this node.
     *
     * Complexity: O(1) */
    incident_triangle_iterator triangle_end() const {
      return incident_triangle_iterator(m_, primal_node_.value().triangles_.end());
    }

  private:
    friend class Mesh<N, E, T>;
    mesh_type * m_;
    typename primal_graph_type::Node primal_node_;

    Node(mesh_type * const m, const typename primal_graph_type::Node & primal_node) : m_(m), primal_node_(primal_node) {
    }
  };

  /** Return the number of nodes in the mesh. */
  size_type size() const {
    //would num_triangles() make more sense?
    return num_nodes();
  }

  /** Synonym for size(). */
  size_type num_nodes() const {
    return primal_.num_nodes();
  }

  /** Return the node with index @a i.
   * @pre 0 <= @a i < num_nodes()
   *
   * Complexity: O(1). */
  Node node(size_type i) const {
    return Node(const_cast<mesh_type *> (this), primal_.node(i));
  }

  /** Add a node to the mesh, returning the added node.
   * @param[in] position The new node's position
   * @param[in] value The new node's value
   * @post new size() == old size() + 1
   * @post The returned node's index() == old size()
   * @post The returned node's value() == @a value
   *
   * Can invalidate outstanding iterators; does not invalidate outstanding
   * Node objects.
   *
   * Complexity: O(1) amortized time. */
  Node add_node(const Point& position,
    const node_value_type& value = node_value_type()) {
    primal_node_value_type pvalue;
    pvalue.value_ = value;
    return Node(const_cast<mesh_type *> (this), primal_.add_node(position, pvalue));
  }

  /** Remove a node from the mesh.
   * @param[in] n node to be removed
   * @pre @a n == node(@a i) for some @a i with 0 <= @a i < size()
   * @post new size() == old size() - 1
   *
   * Can invalidate outstanding iterators. @a n becomes invalid, as do any
   * other Node objects equal to @a n. All other Node objects remain valid.
   *
   * Complexity: Polynomial in size(). */
  void remove_node(Node n) {
    for(incident_triangle_iterator it = n.triangle_begin(); it != n.triangle_end(); it = n.triangle_begin()) {
      Triangle t = *it;
      remove_triangle(t);
    }
    primal_.remove_node(n.primal_node_);
  }
  

  /** Remove all nodes and edges from this mesh.
   * @post size() == 0 && num_edges() == 0 && num_triangles() == 0
   *
   * Invalidates all outstanding iterators, Node, Edge, and Triangle objects. */
  void clear() {
    primal_.clear();
    dual_.clear();
  }


  /* EDGES */

  /** @class Mesh::Edge
   * @brief Class representing the mesh's edges.
   *
   * Edges are order-insensitive pairs of nodes. Two Edges with the same nodes
   * are considered equal if they connect the same nodes, in either order. */
  class Edge {
  public:

    /** Construct an invalid Edge. */
    Edge() {
    }

    /** Returns the first node.
     * @pre this is a valid edge.
     * @return Node n s.t. n != node2() && there is an edge between n and node2().
     *
     * If edge was returned by a node's incident_iterator, returns the node.
     * Otherwise order is undefined.
     *
     * Complexity: O(1) */
    Node node1() const {
      return Node(m_, primal_edge_.node1());
    }

    /** Returns the second node.
     * @pre this is a valid edge.
     * @return Node n s.t. n != node1() && there is an edge between n and node1().
     *
     * If edge was returned by node's incident_iterator, returns the node's neighbor.
     * Otherwise order is undefined.
     *
     * Complexity: O(1) */
    Node node2() const {
      return Node(m_, primal_edge_.node2());
    }

    /** Return the number of triangles sharing this edge.
     *
     * Complexity: O(1) */
    size_t num_triangles() const {
      return primal_edge_.value().triangles_.size();
    }

    /** Test whether this node and @a x are equal.
     * @pre this is a valid edge.
     * @return true iff, for all Nodes n in this Edge, n == n' for some Node in @a x.
     *
     */
    bool operator==(const Edge& x) const {
      return m_ == x.m_ && primal_edge_ == x.primal_edge_;
    }

    /** Return this edge's value. */
    edge_value_type& value() {
      return primal_edge_.value().value_;
    }

    const edge_value_type & value() const {
      return primal_edge_.value().value_;
    }

    double length() const {
      return primal_edge_.length();
    }

  private:
    friend class Mesh<N, E, T>;
    mesh_type * m_;
    typename primal_graph_type::Edge primal_edge_;

    Edge(mesh_type * const m, const typename primal_graph_type::Edge primal_edge) : m_(m), primal_edge_(primal_edge) {
    }
  };

  /** Return the total number of edges.
   *
   * Complexity: O(1). */
  size_type num_edges() const {
    return primal_.num_edges();
  }

  /** Return the edge with index @a i.
   * @pre 0 <= @a i < num_edges()
   *
   * Complexity: O(1). */
  Edge edge(size_type i) const {
    return Edge(const_cast<mesh_type *> (this), primal_.edge(i));
  }

  /** Test whether two nodes are connected by an edge.
   * @pre @a a and @a b are valid nodes of this mesh
   * @return true if, for some @a i, edge(@a i) connects @a a and @a b.
   *
   * Complexity: O(log(num_edges())) */
  bool has_edge(const Node& a, const Node& b) const {
    return primal_.has_edge(a.primal_node_, b.primal_node_);
  }

  /* TRIANGLES */

  /** @class Mesh::Triangle
   * @brief Class representing the mesh's triangles.
   *
   */
  class Triangle {
  public:

    /** Construct an invalid Triangle. */
    Triangle() : m_(NULL), dual_node_() {
    }

    /** Returns the @a ith node.
     * @pre 0 <= i < 3
     * Complexity: O(1) */
    Node node(int i) const {
      return Node(m_, dual_node_.value().nodes_[i]);
    }

    /** Returns the edge between node(i) and node((i+1) % 3)
     * @pre 0 <= i < 3
     * Complexity: O(1) */
    Edge edge(int i) const {
      return Edge(m_, dual_node_.value().edges_[i]);
    }

    /** Test whether this triangle and @a x are equal.
     * @return true iff, for all Nodes n in this Triangle, n == n' for some Node in @a x.
     *
     */
    bool operator==(const Triangle& x) const {
      return m_ == x.m_ && dual_node_ == x.dual_node_;
    }

    /** Return this edge's value. */
    triangle_value_type & value() {
      return dual_node_.value().value_;
    }

    /** The area of this triangle
     * Uses Heron's formula to calculate the area
     * @Pre: the triangle is a valid triangle
     * @return result triangle area
     * Complexity: O(1)
     */
    double area() const {
      Edge e1 = edge(0);
      Edge e2 = edge(1);
      Edge e3 = edge(2);
      double a = e1.primal_edge_.length();
      double b = e2.primal_edge_.length();
      double c = e3.primal_edge_.length();
      double s = (a + b + c) / 2;
      return sqrt(s * (s - a) * (s - b) * (s - c));
    }

    /** The outward normal from this triangle at Edge t.edge(@a i)
     * @pre 0 <= @a i < 3
     * @return unit vector normal to the edge facing away from the triangle
     * Complexity: O(1)
     */
    Point normal(int i) const {
      typename primal_graph_type::Edge e = dual_node_.value().edges_[i];
      //find node that is not e
      typename primal_graph_type::Node n_across = dual_node_.value().nodes_[(i + 2) % 3];

      //let AP be some vector from n_across to the edge (for simplicity lets use one of the two other nodes)
      //using dot product
      Point vector_e = e.node1().position() - e.node2().position();
      Point vector_ap = n_across.position() - e.node2().position();

      vector_e = projection(vector_ap, vector_e);

      return (vector_e - vector_ap).normalize();
    }

    /** Returns the normal to this surface, where the orientation is given
     * by taking the points of this triangle in counter-clockwise (geometrically
     * and not by index). O(1) */
    Point normal() const {
      Point a = node(0).position();
      Point b = node(1).position();
      Point c = node(2).position();
      Point u = b - a;
      Point v = c - a;
      Point normal = u.cross(v);
      return (determinant(a, b, c) > 0) ? normal : normal * -1;
    }

    /** The number of neighboring triangles
     *  @pre triangle node is valid
     *  @result return the number of adjacent triangles to this triangle
     *  Complexity: O(1)
     */
    size_t num_neighbors() const {
      return dual_node_.degree();
    }

    /** Return the index of the triangle
     * @pre: triangle is valid
     * @return: triangle's index
     * Complexity: O(1)
     */
    size_type index() const {
      return dual_node_.index();
    }

    /** True if @a t shares only 1 edge with this triangle
     *  @pre: both @a t and this triangle is valid
     *  @return: true if and only if t shares and edge with this
     *  Complexity: O(1)
     **/
    bool is_adjacent(Triangle t) const {
      int numAdjacentCounter = 0;
      for (int i = 0; i < 3; ++i) {
        if (node(i) == t.node(0))
          numAdjacentCounter++;
        else if (node(i) == t.node(1))
          numAdjacentCounter++;
        else if (node(i) == t.node(2))
          numAdjacentCounter++;
      }
      return numAdjacentCounter == 2;
    }

    /** Returns an iterator referring to the first triangle incident to this node.
     *
     * Complexity: O(1) */
    triangle_neighbor_iterator triangle_begin() const {
      return triangle_neighbor_iterator(m_, dual_node_.edge_begin());
    }

    /** Returns an iterator referring to the past-the-end triangle incident to this node.
     *
     * Complexity: O(1) */
    triangle_neighbor_iterator triangle_end() const {
      return triangle_neighbor_iterator(m_, dual_node_.edge_end());
    }


  private:
    friend class Mesh<N, E, T>;

    /** Computes the determinant of the matrix where the rows
     * are the entries of a, b, c respectively. */
    double determinant(Point a, Point b, Point c) const {
      return a.dot(b.cross(c));
    }


    mesh_type * m_;
    typename dual_graph_type::Node dual_node_;

    Triangle(mesh_type * const m, const typename dual_graph_type::Node & dual_node) : m_(m), dual_node_(dual_node) {
    }

    Triangle(mesh_type * const m, const typename dual_graph_type::Edge & incoming_dual_edge) : m_(m), dual_node_(incoming_dual_edge.node2()) {
    }
  };

  /** Return the total number of triangles.
   * Complexity: O(1). */
  size_type num_triangles() const {
    return dual_.num_nodes();
  }

  /** Return the triangle with index @a i.
   * @pre 0 <= @a i < num_edges()
   *
   * Complexity: O(1). */
  Triangle triangle(size_type i) const {
    return Triangle(const_cast<mesh_type *> (this), dual_.node(i));
  }

  /** Test whether three nodes share a triangle.
   * @pre @a a, @a b, and @a c are valid nodes of this graph
   * @return true if, for some @a i, triangle(@a i) connects @a a, @a b, and @a c.
   *
   * Complexity: O(min(a.num_triangles(), b.num_triangles(), c.num_triangles())) */
  bool has_triangle(const Node& a, const Node& b, const Node& c) const {
    typename primal_graph_type::Node nodes[3];
    nodes[0] = a.primal_node_;
    nodes[1] = b.primal_node_;
    nodes[2] = c.primal_node_;
    sort_nodes(nodes);

    //find node with smallest number of triangles to check
    unsigned int num_triangles = nodes[0].value().triangles_.size();
    int s = 0;
    for (int i = 1; i < 3; i++) {
      if (nodes[i].value().triangles_.size() < num_triangles) {
        s = i;
        num_triangles = nodes[i].value().triangles_.size();
      }
    }

    //since nodes are sorted in triangle in ascending order, if the triangle exists,
    //it's "nodes_" array will be the same as "nodes"
    for (typename dual_node_list_type::const_iterator it = nodes[s].value().triangles_.begin(); it != nodes[s].value().triangles_.end(); ++it) {
      const typename dual_graph_type::Node & t = *it;
      if (t.value().nodes_[s] == nodes[s] &&
        t.value().nodes_[(s + 1) % 3] == nodes[(s + 1) % 3] &&
        t.value().nodes_[(s + 2) % 3] == nodes[(s + 2) % 3]) {
        return true;
      }
    }
    return false;
  }

  /** Removes a triangle from the Mesh
   *  @param[in] t The triangle to be removed 
   *  @pre the mesh has triangle t
   *  @post The mesh no longer has triangle t
   *  @post The mesh no longer has an edge between node n1 and n2 if there is no triangle with edge(n1,n2)
   *  @post new num_triangles() = old num_triangles() - 1
   *  @post outstanding iterators can be invalidated
   **/
  void remove_triangle(Triangle &t){

    //remove this ref from every one of triangles nodes
    for(int i=0; i<3; i++) {
      Node n = t.node(i);

      //get the iterators for the triangle associated with a node
      auto tri_list_begin = n.primal_node_.value().triangles_.begin();
      auto tri_list_end = n.primal_node_.value().triangles_.end();

      //find its position in the list
      auto t_itpos = find(tri_list_begin,tri_list_end,t.dual_node_);
      assert(t_itpos != tri_list_end);

      //remove the tri from the list
      n.primal_node_.value().triangles_.erase(t_itpos);

    }



    //remove this ref from every one of triangles edges
    for(int i=0; i<3; i++) {
      Edge e = t.edge(i);

      //get the iterators for the triangle associated with an edge
      auto tri_list_begin = e.primal_edge_.value().triangles_.begin();
      auto tri_list_end = e.primal_edge_.value().triangles_.end();

      //find its pos in the list
      auto t_itpos = find(tri_list_begin,tri_list_end,t.dual_node_);
      assert(t_itpos != tri_list_end);

      //remove it from the list
      e.primal_edge_.value().triangles_.erase(t_itpos);

      //remove edge if e has no triangles left
      if(e.primal_edge_.value().triangles_.size()==0)
        primal_.remove_edge(e.primal_edge_);
    }


    //remove this tri from dual graph
    dual_.remove_node(t.dual_node_);


  }

  /** Add a triangle to the graph, or return the current triangle if one exists.
   * @pre @a a, @a b, and @a c are valid nodes of this graph
   * @pre @a a!=@a b, @a b!=@a c, @a a!=@a c
   * @pre there currently is no triangle containing @a a, @a b, and @a c
   * @return a Triangle object t, where for n in (@a a, @a b, @a c) there exists
   * an i, 0<=i<3, s.t. t.node(i)==n.  Nodes are sorted in ascending order.
   * @post has_triangle(@a a, @a b, @a c) == true //TODO
   * @post If old has_triangle(@a a, @a b, @a c) == true, then new num_triangles() ==
   *   old num_triangles(). Otherwise, new num_triangles() == old num_triangles() + 1. //TODO
   *
   * Can invalidate all triangle, edge and incident iterators.
   * Must not invalidate outstanding Triangle objects.
   *
   * Complexity: O(num_edges()) */
  Triangle add_triangle(const Node& a, const Node& b, const Node& c,
    const triangle_value_type& value = triangle_value_type()) {

    triangle_info t_v;
    t_v.value_ = value;

    //STEP 1: Create mappings from triangle to nodes/edges
    //STEP 1A: prepare nodes, so triangle has them sorted by index
    //add edges and nodes to triangle
    t_v.nodes_[0] = a.primal_node_;
    t_v.nodes_[1] = b.primal_node_;
    t_v.nodes_[2] = c.primal_node_;
//    sort_nodes(t_v.nodes_);
//    std::swap(t_v.nodes_[0], t_v.nodes_[2]);
    //implicitly asserts precondition that nodes are not equal
//    assert(t_v.nodes_[0] < t_v.nodes_[1] && t_v.nodes_[1] < t_v.nodes_[2]);

    //STEP 1B: update primal graph, so we have edges for triangle

    //create triangle's edges in primal
    for (int i = 0; i < 3; ++i) {
      t_v.edges_[i] = primal_.add_edge(t_v.nodes_[i], t_v.nodes_[(i + 1) % 3]);
    }

    //STEP 1C: add triangle to dual
    typename dual_graph_type::Node t = dual_.add_node(t_v);

    //STEP 2: Create mappings from nodes/edges to triangle
    for (int i = 0; i < 3; ++i) {
      //add triangle to node i
      t_v.nodes_[i].value().triangles_.push_back(t);

      dual_node_list_type & triangles = t_v.edges_[i].value().triangles_;
      //link triangle to all triangles on which this edge is present
      dual_edge_info d;
      d.primal_edge_ = t_v.edges_[i];
      for (typename dual_node_list_type::iterator it = triangles.begin(); it != triangles.end(); ++it) {
        //this dual edge refers to it's primal counterpart
        dual_.add_edge(t, *it, d);
      }
      //add triangle to edge i
      triangles.push_back(t);
    }

    return Triangle(const_cast<mesh_type *> (this), t);
  }

  //  /** Remove a triangle.
  //   * @pre @a t is a valid triangle of this graph
  //   * @pre has_triangle(@a t.node1(), @a t.node2(), @a t.node3()) == true
  //   * @post has_edge(@a t.node1(), @a t.node2(), @a t.node3()) == false
  //   * @post new num_triangles() == old num_triangles() - 1
  //   *
  //   * Can invalidate all triangle, edge and incident iterators.
  //   * Must not invalidate outstanding Triangle objects that do not equal @a t.
  //   *
  //   * Complexity: O(num_edges()) */
  //  void remove_triangle(const Triangle& t) {
  //  }

  /* ITERATORS */

  /** @class Mesh::base_iterator
   * @brief Iterator template for node/edge/triangle iterators. A forward iterator. */
  template <typename IT, typename V>
  class base_iterator {
  public:
    /** Element type. */
    typedef V value_type;
    /** Type of pointers to elements. */
    typedef V* pointer;
    /** Type of references to elements. */
    typedef V& reference;
    /** Iterator category. */
    typedef std::input_iterator_tag iterator_category;

    /** Construct an invalid iterator. */
    base_iterator() : m_(NULL), it_() {
    }

    /** Return the value to which this iterator points.
     * Complexity: O(1) */
    V operator*() const {
      return V(m_, *it_);
    }

    /** Increment this iterator
     * Complexity: O(1) */
    base_iterator& operator++() {
      ++it_;
      return *this;
    }

    /** Test whether this iterator and @a x are equal.
     * @pre both iterators are for the same mesh
     *
     * this iterator == @a x iff both refer to the same node or both are
     * past-the-end.
     *
     * Complexity: O(1) */
    bool operator==(const base_iterator<IT, V>& x) const {
      return it_ == x.it_;
    }

    /** Test whether this iterator is less than @a x.
     * @pre both iterators are for the same mesh
     *
     * this iterator < @a x iff this iterator refers to a node that has a smaller
     * index than the node refered to by @a x, or if this iterator refers to a
     * node and @a x is past-the-end.
     *
     * Complexity: O(1) */
    bool operator<(const base_iterator<IT, V>& x) const {
      return it_ < x.it_;
    }

  protected:
    friend class Mesh<N, E, T>;
    mesh_type * m_;
    IT it_;

    base_iterator(mesh_type * const m, const IT it) : m_(m), it_(it) {
    }

  };

  class triangle_neighbor_iterator : public base_iterator<typename dual_graph_type::incident_iterator, Triangle> {
  public:

    /** Return common edge of current triangle
     * Complexity: O(1) */
    Edge edge() const {
      return Edge(super::m_, (*super::it_).value().primal_edge_);
    }

    triangle_neighbor_iterator(mesh_type * const m, const typename dual_graph_type::incident_iterator it)
      : super(m, it) {
    }
  private:
    typedef base_iterator<typename dual_graph_type::incident_iterator, Triangle> super;
    friend class Mesh<N, E, T>;
  };

  /** Returns an iterator referring to the first node in the mesh.
   *
   * Complexity: O(1) */
  node_iterator node_begin() const {
    return node_iterator(const_cast<mesh_type *> (this), primal_.node_begin());
  }

  /** Returns an iterator referring to the past-the-end node in the mesh.
   *
   * Complexity: O(1) */
  node_iterator node_end() const {
    return node_iterator(const_cast<mesh_type *> (this), primal_.node_end());
  }

  /** Returns an iterator referring to the first edge in the mesh.
   *
   * Complexity: O(1) */
  edge_iterator edge_begin() const {
    return edge_iterator(const_cast<mesh_type *> (this), primal_.edge_begin());
  }

  /** Returns an iterator referring to the past-the-end edge in the mesh.
   *
   * Complexity: O(1) */
  edge_iterator edge_end() const {
    return edge_iterator(const_cast<mesh_type *> (this), primal_.edge_end());
  }

  /** Returns an iterator referring to the first triangle in the mesh.
   *
   * Complexity: O(1) */
  triangle_iterator triangle_begin() const {
    return triangle_iterator(const_cast<mesh_type *> (this), dual_.node_begin());
  }

  /** Returns an iterator referring to the past-the-end triangle in the mesh.
   *
   * Complexity: O(1) */
  triangle_iterator triangle_end() const {
    return triangle_iterator(const_cast<mesh_type *> (this), dual_.node_end());
  }

  neighborhood_iterator node_begin(const BoundingBox& b) const {
    return neighborhood_iterator(const_cast<mesh_type *> (this), primal_.node_begin(b));
  }

  neighborhood_iterator node_end(const BoundingBox& b) const {
    return neighborhood_iterator(const_cast<mesh_type *> (this), primal_.node_end(b));
  }

  //  typedef base_iterator<typename primal_graph_type::incident_iterator, Edge> incident_iterator;

private:

  inline void sort_nodes(typename primal_graph_type::Node nodes[3]) const {
    //basically, a hard-coded bubble sort
    if (nodes[0] > nodes[1]) {
      std::swap(nodes[0], nodes[1]);
    }
    if (nodes[1] > nodes[2]) {
      std::swap(nodes[1], nodes[2]);
    }
    if (nodes[0] > nodes[1]) {
      std::swap(nodes[0], nodes[1]);
    }
    assert(nodes[0] <= nodes[1] && nodes[1] <= nodes[2]);
  }

  primal_graph_type primal_;
  dual_graph_type dual_;
};

#endif
