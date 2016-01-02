#ifndef CS207_GRAPH_HPP
#define CS207_GRAPH_HPP

/**
 * @file Graph.hpp
 */

#include "CS207/Util.hpp"
#include <utility>
#include <algorithm>
#include <vector>
#include <map>
#include <assert.h>

// Automatically derive !=, <=, >, and >= from a class's == and <
using namespace std::rel_ops;

/** @class Graph
 * @brief A template for 3D undirected graphs.
 *
 * Users can add and retrieve nodes and edges. Edges are unique (there is at
 * most one edge between any pair of nodes). */
template <typename N, typename E>
class Graph {
  // Internal type for node elements
  struct node_info;
  typedef struct node_info node_info_type;

  // Internal type for edge elements
  struct edge_info;
  typedef struct edge_info edge_info_type;

  /** Type of node unique identifiers */
  typedef unsigned node_id_type;
  /** Type of edge unique identifiers */
  typedef unsigned edge_id_type;

  /** adjacency list types */
  struct neighbor;
  typedef struct neighbor adjacency_type;
  typedef std::vector<adjacency_type> adjacency_list_type;


public:

  // PUBLIC TYPE DEFINITIONS

  struct PerfStats {
    //rough estimate of memory usage for nodes;
    unsigned node_mem_count_;
    //rough estimate of memory usage for nodes;
    unsigned edge_mem_count_;

    PerfStats() : node_mem_count_(0), edge_mem_count_(0) {

    }
  };
  /** Synonym for PerfStats (following STL conventions). */
  typedef PerfStats perf_stats_type;

  /** Type of this graph. */
  typedef Graph<N, E> graph_type;

  /** Type of graph nodes. */
  class Node;
  /** Synonym for Node (following STL conventions). */
  typedef Node node_type;

  /** Type of graph edges. */
  class Edge;
  /** Synonym for Edge (following STL conventions). */
  typedef Edge edge_type;

  /** Type of node values (HW1 #4 and beyond). */
  typedef N node_value_type;
  typedef E edge_value_type;


  /** Type of indexes and sizes. Return type of Node::index() and
      Graph::num_nodes(), argument type of Graph::node(). */
  typedef unsigned size_type;

  /* base type for most iterators */
  template <typename IT, typename VAL>
  class base_iterator;

  /** Type of node iterators. */
  typedef base_iterator<typename std::vector<node_id_type>::const_iterator, Node> node_iterator;

  /** Type of edge iterators, which iterate over all graph edges. */
  typedef base_iterator<typename std::vector<edge_id_type>::const_iterator, Edge> edge_iterator;

  /** Type of incident iterators, which iterate over the edges that touch
      a given node. */
  class incident_iterator;


  // CONSTRUCTOR AND DESTRUCTOR

  /** Construct an empty graph. */
  Graph() :
    nodes_(), edges_(), idx2uid_(), eidx2uid_(), free_node_(INVALID_NODE), free_edge_(INVALID_EDGE), stats_() {
  }

  /** Destroy the graph. */
  ~Graph() {
  }


  // NODES

  /** @class Graph::Node
   * @brief Class representing the graph's nodes.
   *
   * Node objects are used to access information about the Graph's nodes.
   * The graph offers several ways to access nodes, including iterators
   * and the node() function.
   */
  class Node {
  public:

    /** Construct an invalid node.
     *
     * Valid nodes are obtained from the Graph class or its iterators, but it
     * is occasionally useful to declare an @i invalid node, and assign a
     * valid node to it later. For example:
     *
     * @code
     * Node x;
     * if (...should pick the first node...)
     *   x = graph.node(0);
     * else {
     *   // pick some other node using a complicated calculation
     * }
     * @endcode */
    Node() : g_(NULL), uid_(INVALID_NODE), gen_(INVALID_GEN) {
    }

    /** Return the graph that contains this node. */
    graph_type* graph() const {
      return g_;
    }

    /** Return this node's value. */
    const node_value_type & value() const {
      assert(g_->nodes_[uid_].gen_ == gen_);
      return g_->nodes_[uid_].v_;
    }

    node_value_type & value() {
      assert(g_->nodes_[uid_].gen_ == gen_);
      return g_->nodes_[uid_].v_;
    }

    /** Return this node's index, a number in the range [0, graph_size). */
    size_type index() const {
      assert(g_->nodes_[uid_].gen_ == gen_);
      return g_->nodes_[uid_].idx_;
    }

    /** Test whether this node and @a x are equal.
     *
     * Equal nodes have the same graph and the same index. */
    bool operator==(const Node& x) const {
      return (g_ == x.g_) && (index() == x.index());
    }

    /** Test whether this node is less than @a x in the global order.
     *
     * This ordering function is useful for STL containers such as
     * std::map<>. It need not have any geometric meaning.
     *
     * The node ordering relation must obey trichotomy: For any two nodes x
     * and y, exactly one of x == y, x < y, and y < x is true. */
    bool operator<(const Node& x) const {
      return (g_ < x.g_) || (g_ == x.g_ && index() < x.index());
    }

    /** Return the number of edges incident to this node.
     *
     * Complexity: O(1) */
    size_t degree() const {
      assert(g_->nodes_[uid_].gen_ == gen_);
      return g_->nodes_[uid_].adjacency_.size();
    }

    /** Returns an iterator referring to the first edge incident to this node.
     *
     * Complexity: O(1) */
    incident_iterator edge_begin() const {
      assert(g_->nodes_[uid_].gen_ == gen_);
      return incident_iterator(g_, uid_, g_->nodes_[uid_].adjacency_.begin());
    }

    /** Returns an iterator referring to the past-the-end edge incident to this node.
     *
     * Complexity: O(1) */
    incident_iterator edge_end() const {
      assert(g_->nodes_[uid_].gen_ == gen_);
      return incident_iterator(g_, uid_, g_->nodes_[uid_].adjacency_.end());
    }

  private:
    friend class Graph<N, E>;
    graph_type* g_;
    node_id_type uid_;
    size_type gen_;

    Node(graph_type * const g, node_id_type uid) : g_(g), uid_(uid), gen_(g->nodes_[uid].gen_) {
    }
  };

  /** Return performance statistics on this graph */
  const PerfStats & stats() const {
    return stats_;
  }

  /** Reset performance statistics on this graph */
  void reset_stats() {
    stats_ = perf_stats_type();
  }

  /** Return the number of nodes in the graph. */
  size_type size() const {
    return idx2uid_.size();
  }

  /** Synonym for size(). */
  size_type num_nodes() const {
    return size();
  }

  /** Return the node with index @a i.
   * @pre 0 <= @a i < num_nodes()
   *
   * Complexity: O(1). */
  Node node(size_type i) const {
    assert(i < num_nodes()); //i is unsigned, always >= 0
    return Node(const_cast<graph_type *> (this), idx2uid_[i]);
  }

  /** Add a node to the graph, returning the added node.
   * @param[in] value The new node's value
   * @post new size() == old size() + 1
   * @post The returned node's index() == old size()
   * @post The returned node's value() == @a value
   *
   * Can invalidate outstanding iterators; does not invalidate outstanding
   * Node objects.
   *
   * Complexity: O(1) amortized time. */
  Node add_node(const node_value_type& value = node_value_type()) {
    node_id_type id = alloc_node();
    nodes_[id].v_ = value;
    nodes_[id].idx_ = idx2uid_.size();

    idx2uid_.push_back(id);

    return Node(const_cast<graph_type *> (this), id);
  }

  /** Remove a node from the graph.
   * @param[in] n node to be removed
   * @pre @a n == node(@a i) for some @a i with 0 <= @a i < size()
   * @post new size() == old size() - 1
   *
   * Can invalidate outstanding iterators. @a n becomes invalid, as do any
   * other Node objects equal to @a n. All other Node objects remain valid.
   *
   * Complexity: Polynomial in size(). */
  void remove_node(Node n) {

    //remove edges from neighbors
    adjacency_list_type & neighbors = nodes_[n.uid_].adjacency_;
    typename adjacency_list_type::iterator e_it;
    for (e_it = neighbors.begin(); e_it < neighbors.end(); ++e_it) {
      adjacency_list_type & neigh_edges = nodes_[(*e_it).nid_].adjacency_;
      typename adjacency_list_type::iterator neigh_it = lower_bound(neigh_edges.begin(), neigh_edges.end(), n.uid_);
      assert((*neigh_it) == n.uid_);
      remove_edge_idx((*neigh_it).eid_);
      free_edge((*neigh_it).eid_);
      neigh_edges.erase(neigh_it);
    }

    //update indices of nodes with higher indices
    remove_node_idx(n.uid_);
    free_node(n.uid_);

  }

  /** Remove all nodes and edges from this graph.
   * @post size() == 0 && num_edges() == 0
   *
   * Invalidates all outstanding iterators and Node objects. */
  void clear() {
    for (int i = size() - 1; i >= 0; --i) {
      //remove last node.  since adj vectors are sorted by node id, this will prevent
      //any vector element shifting
      remove_node(node(i));
    }
    stats_ = perf_stats_type();

    assert(idx2uid_.size() == 0);
  }


  // EDGES

  /** @class Graph::Edge
   * @brief Class representing the graph's edges.
   *
   * Edges are order-insensitive pairs of nodes. Two Edges with the same nodes
   * are considered equal if they connect the same nodes, in either order. */
  class Edge {
  public:

    /** Construct an invalid Edge. */
    Edge() : g_(NULL), uid_(INVALID_EDGE), gen_(INVALID_GEN) {
    }

    /** Returns the first node.
     * @pre this is a valid edge.
     * @return Node n s.t. n != node2() && there is an edge between n and node2().
     *
     * If edge was returned by a node's incident_iterator, returns the node.
     * Order is undefined otherwise.
     *
     * Complexity: O(1) */
    Node node1() const {
      return Node(g_, g_->edges_[uid_].n1_);
    }

    /** Returns the second node.
     * @pre this is a valid edge.
     * @return Node n s.t. n != node1() && there is an edge between n and node1().
     *
     * If edge was returned by node's incident_iterator, returns the node's neighbor.
     * Order is undefined otherwise.
     *
     * Complexity: O(1) */
    Node node2() const {
      return Node(g_, g_->edges_[uid_].n2_);
    }

    /** Test whether this node and @a x are equal.
     * @pre this is a valid edge.
     * @return true iff, for all Nodes n in this Edge, n == n' for some Node in @a x.
     *
     */
    bool operator==(const Edge& x) const {
      //Depends on the invariant that two nodes have at most one edge between them
      return x.uid_ == uid_;
    }

    /** Return this edge's value. */
    const edge_value_type & value() const {
      assert(g_->edges_[uid_].gen_ == gen_);
      return g_->edges_[uid_].v_;
    }

    edge_value_type & value() {
      assert(g_->edges_[uid_].gen_ == gen_);
      return g_->edges_[uid_].v_;
    }

  private:
    friend class Graph<N, E>;
    graph_type * g_;
    edge_id_type uid_;
    size_type gen_;

    Edge(graph_type * const g, edge_id_type uid)
      : g_(g), uid_(uid), gen_(g->edges_[uid].gen_) {
      assert(node1() != node2());
    }
  };

  /** Return the total number of edges.
   *
   * Complexity: O(1). */
  size_type num_edges() const {
    return eidx2uid_.size();
  }

  /** Return the edge with index @a i.
   * @pre 0 <= @a i < num_edges()
   *
   * Complexity: O(1). */
  Edge edge(size_type i) const {
    return Edge(const_cast<graph_type *> (this), eidx2uid_[i]);
  }

  /** Test whether two nodes are connected by an edge.
   * @pre @a a and @a b are valid nodes of this graph
   * @return true if, for some @a i, edge(@a i) connects @a a and @a b.
   *
   * Complexity: O(log(num_edges())) */
  bool has_edge(const Node& a, const Node& b) const {
    //index() checks validity of nodes
    const adjacency_list_type & a_edges = nodes_[a.uid_].adjacency_;
    typename adjacency_list_type::const_iterator a_it = lower_bound(a_edges.begin(), a_edges.end(), b.uid_);
    return (a_it != a_edges.end() && *a_it == b.uid_);
  }

  /** Add an edge to the graph, or return the current edge if one exists.
   * @pre @a a and @a b are valid nodes of this graph, and a != b
   * @return an Edge object e with e.node1() == @a a and e.node2() == @a b
   * @post has_edge(@a a, @a b) == true
   * @post If old has_edge(@a a, @a b) == true, then new num_edges() ==
   *   old num_edges(). Otherwise, new num_edges() == old num_edges() + 1.
   *
   * Can invalidate edge indexes -- in other words, old edge(@a i) might not
   * equal new edge(@a i). Can invalidate all edge and incident iterators.
   * Must not invalidate outstanding Edge objects.
   *
   * Complexity: O(num_edges()) */
  Edge add_edge(const Node& a, const Node& b, const edge_value_type& value = edge_value_type()) {
    //'==' checks validity of nodes
    assert(a != b);
    adjacency_list_type & a_edges = nodes_[a.uid_].adjacency_;
    adjacency_list_type & b_edges = nodes_[b.uid_].adjacency_;
    typename adjacency_list_type::iterator a_it = lower_bound(a_edges.begin(), a_edges.end(), b.uid_);
    typename adjacency_list_type::iterator b_it = lower_bound(b_edges.begin(), b_edges.end(), a.uid_);

    edge_id_type eid;
    if (!(a_it != a_edges.end() && *a_it == b.uid_)) {
      //alloc edge
      eid = alloc_edge();
      edges_[eid].n1_ = a.uid_;
      edges_[eid].n2_ = b.uid_;
      edges_[eid].v_ = value;
      edges_[eid].idx_ = eidx2uid_.size();

      eidx2uid_.push_back(eid);

      a_edges.insert(a_it, adjacency_type(eid, b.uid_));
      b_edges.insert(b_it, adjacency_type(eid, a.uid_));
    } else {
      eid = (*a_it).eid_;
    }

    return Edge(const_cast<graph_type *> (this), eid);
  }

  /** Remove an edge, if any, returning the number of edges removed.
   * @pre @a a and @a b are valid nodes of this graph
   * @return 0 if old has_edge(@a a, @a b) == false, 1 otherwise
   * @post !has_edge(@a a, @a b)
   * @post new num_edges() == old num_edges() - return value
   *
   * Can invalidate edge indexes -- in other words, old edge(@a i) might not
   * equal new edge(@a i). Can invalidate all edge and incident iterators.
   * Invalidates any edges equal to Edge(@a a, @a b). Must not invalidate
   * other outstanding Edge objects.
   *
   * Complexity: O(num_edges()) */
  size_type remove_edge(const Node& a, const Node& b) {
    adjacency_list_type & a_edges = nodes_[a.uid_].adjacency_;
    adjacency_list_type & b_edges = nodes_[b.uid_].adjacency_;
    typename adjacency_list_type::iterator a_it = lower_bound(a_edges.begin(), a_edges.end(), b.uid_);
    typename adjacency_list_type::iterator b_it = lower_bound(b_edges.begin(), b_edges.end(), a.uid_);

    if (a_it != a_edges.end() && *a_it == b.uid_) {
      remove_edge_idx((*a_it).eid_);
      free_edge((*a_it).eid_);
      a_edges.erase(a_it);
      b_edges.erase(b_it);

      return 1;
    }
    return 0;
  }

  /** Remove an edge.
   * @pre @a e is a valid edge of this graph
   * @pre has_edge(@a e.node1(), @a e.node2()) == true
   * @post has_edge(@a e.node1(), @a e.node2()) == false
   * @post new num_edges() == old num_edges() - 1
   *
   * This is a synonym for remove_edge(@a e.node1(), @a e.node2()), but its
   * implementation can assume that @a e is definitely an edge of the graph.
   * This might allow a faster implementation.
   *
   * Can invalidate edge indexes -- in other words, old edge(@a i) might not
   * equal new edge(@a i). Can invalidate all edge and incident iterators.
   * Invalidates any edges equal to Edge(@a a, @a b). Must not invalidate
   * other outstanding Edge objects.
   *
   * Complexity: O(num_edges()) */
  void remove_edge(const Edge& e) {
    int i = remove_edge(e.node1(), e.node2());
    (void) i;
    assert(i == 1);
  }


  // ITERATORS

  /** @class Graph::base_iterator
   * @brief Iterator template for node/edge iterators. A forward iterator. */
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
    base_iterator() : g_(NULL), it_() {
    }

    /** Return the value to which this iterator points.
     * Complexity: O(1) */
    V operator*() const {
      return V(g_, *it_);
    }

    /** Increment this iterator
     * Complexity: O(1) */
    base_iterator& operator++() {
      ++it_;
      return *this;
    }

    /** Test whether this iterator and @a x are equal.
     * @pre both iterators are for the same graph
     *
     * this iterator == @a x iff both refer to the same node or both are
     * past-the-end.
     *
     * Complexity: O(1) */
    bool operator==(const base_iterator<IT, V>& x) const {
      return it_ == x.it_;
    }

    /** Test whether this iterator is less than @a x.
     * @pre both iterators are for the same graph
     *
     * this iterator < @a x iff this iterator refers to a node that has a smaller
     * index than the node refered to by @a x, or if this iterator refers to a
     * node and @a x is past-the-end.
     *
     * Complexity: O(1) */
    bool operator<(const base_iterator<IT, V>& x) const {
      return it_ < x.it_;
    }

  private:
    friend class Graph<N, E>;
    graph_type * g_;
    IT it_;

    base_iterator(graph_type * const g, const IT it) : g_(g), it_(it) {
    }

  };

  /** Returns an iterator referring to the first node in the graph.
   *
   * Complexity: O(1) */
  node_iterator node_begin() const {

    return node_iterator(const_cast<graph_type *> (this), idx2uid_.begin());
  }

  /** Returns an iterator referring to the past-the-end node in the graph.
   *
   * Complexity: O(1) */
  node_iterator node_end() const {

    return node_iterator(const_cast<graph_type *> (this), idx2uid_.end());
  }

  /** Returns an iterator referring to the first edge in the graph.
   *
   * Complexity: O(1) */
  edge_iterator edge_begin() const {

    return edge_iterator(const_cast<graph_type *> (this), eidx2uid_.begin());
  }

  /** Returns an iterator referring to the past-the-end edge in the graph.
   *
   * Complexity: O(1) */
  edge_iterator edge_end() const {

    return edge_iterator(const_cast<graph_type *> (this), eidx2uid_.end());
  }

  /** @class Graph::incident_iterator
   * @brief Iterator class for edges incident to a given node. A forward
   * iterator. */
  class incident_iterator {
  public:
    // These type definitions help us use STL's iterator_traits.

    /** Element type. */
    typedef Edge value_type;
    /** Type of pointers to elements. */
    typedef Edge* pointer;
    /** Type of references to elements. */
    typedef Edge& reference;
    /** Iterator category. */
    typedef std::input_iterator_tag iterator_category;

    /** Construct an invalid incident_iterator. */
    incident_iterator() : g_(NULL), e_it_() {
    }

    /** Dereference this iterator.
     * @pre this iterator < Node::edge_end()
     * @return an Edge e where e.node1() is the iterator's node.
     *
     * Complexity: O(1) */
    Edge operator*() const {
      //to make Edge's life easier
      if (g_->edges_[(*e_it_).eid_].n1_ != node_uid_) {
        std::swap(g_->edges_[(*e_it_).eid_].n1_, g_->edges_[(*e_it_).eid_].n2_);
      }
      return Edge(g_, (*e_it_).eid_);
    }

    /** Increment this iterator
     * @pre this iterator < Node::edge_end()
     * @post if old *iterator refers to edge(i), new *iterator refers to
     * edge(i+1).  If edge(i) is the last incident edge, new iterator is past-the-end
     *
     * Complexity: O(1) */
    incident_iterator& operator++() {
      ++e_it_;
      return *this;
    }

    /** Test whether this iterator and @a x are equal.
     * @pre both iterators are for the same node
     *
     * this iterator == @a x iff both refer to the same edge or both are
     * past-the-end.
     *
     * Complexity: O(1) */
    bool operator==(const incident_iterator& x) const {

      return e_it_ == x.e_it_;
    }

    /** Test whether this iterator is less than @a x.
     * @pre both iterators are for the same node
     *
     * this iterator < @a x iff this iterator refers to an edge that has a smaller
     * neighbor uid than the edge refered to by @a x, or if this iterator refers to an
     * edge and @a x is past-the-end.
     *
     * Complexity: O(1) */
    bool operator<(const incident_iterator& x) const {

      return e_it_ < x.e_it_;
    }

  private:
    friend class Graph<N, E>;

    graph_type * g_;
    node_id_type node_uid_;
    typename adjacency_list_type::const_iterator e_it_;

    incident_iterator(graph_type * g, size_type node_uid, typename adjacency_list_type::const_iterator e_it)
      : g_(g), node_uid_(node_uid), e_it_(e_it) {
    }

  };

private:

  static const node_id_type INVALID_NODE = 0xffffffff;
  static const edge_id_type INVALID_EDGE = 0xffffffff;
  static const size_type INVALID_GEN = 0xffffffff;
  static const size_type STARTING_GEN = 0;

  struct node_info {
    node_value_type v_;

    union {
      size_type idx_;
      node_id_type free_;
    };
    size_type gen_;
    adjacency_list_type adjacency_;

    node_info() : v_(), idx_(), gen_(STARTING_GEN), adjacency_() {

    }
  };

  struct edge_info {
    edge_value_type v_;

    union {
      size_type idx_;
      edge_id_type free_;
    };
    node_id_type n1_;
    node_id_type n2_;
    size_type gen_;

    edge_info() : v_(), idx_(), n1_(), n2_(), gen_(STARTING_GEN) {

    }
  };

  struct neighbor {
    edge_id_type eid_;
    node_id_type nid_;

    neighbor(const edge_id_type edge, const node_id_type node)
      : eid_(edge), nid_(node) {

    }

    bool operator<(const node_id_type node) const {
      return nid_ < node;
    }

    bool operator==(const node_id_type node) const {
      return nid_ == node;
    }

    bool operator>(const node_id_type node) const {
      return nid_ > node;
    }
  };

  node_id_type alloc_node() {
    node_id_type id;
    if (free_node_ != INVALID_NODE) {
      id = free_node_;
      free_node_ = nodes_[free_node_].free_;
    } else {
      id = nodes_.size();
      nodes_.push_back(node_info_type());
      stats_.node_mem_count_ = nodes_.size();
    }
    return id;
  }

  void free_node(node_id_type id) {
    nodes_[id].v_ = node_value_type();
    nodes_[id].free_ = free_node_;
    nodes_[id].gen_++;
    nodes_[id].adjacency_.clear();
    free_node_ = id;
  }

  void remove_node_idx(node_id_type id) {
    remove_idx(idx2uid_, nodes_, id);
  }

  edge_id_type alloc_edge() {
    edge_id_type id;
    if (free_edge_ != INVALID_EDGE) {
      id = free_edge_;
      free_edge_ = edges_[free_edge_].free_;
    } else {
      id = edges_.size();
      edges_.push_back(edge_info_type());
      stats_.edge_mem_count_ = edges_.size();
    }
    return id;
  }

  void free_edge(edge_id_type id) {
    edges_[id].v_ = edge_value_type();
    edges_[id].free_ = free_edge_;
    edges_[id].gen_++;
    free_edge_ = id;
  }

  void remove_edge_idx(edge_id_type id) {
    remove_idx(eidx2uid_, edges_, id);
  }

  template <typename IDXS, typename VEC, typename ID>
  void remove_idx(IDXS & idxs, VEC & vec, ID uid) {
    typename IDXS::iterator it = idxs.end();
    typename IDXS::iterator del_it = idxs.begin() + vec[uid].idx_;
    assert(del_it < it);
    for (--it; it > del_it; --it) {
      --(vec[*it].idx_);
    }
    idxs.erase(del_it);
  }

  //list of node_info structs.
  std::vector<node_info_type> nodes_;
  //list of edge_info structs.
  std::vector<edge_info_type> edges_;
  //index is node index (node_info_type.idx_), value is index into nodes_
  std::vector<node_id_type> idx2uid_;
  //index is edge index (edge_info_type.idx_), value is index into edges_
  std::vector<edge_id_type> eidx2uid_;
  //a node uid value that is free to use
  node_id_type free_node_;
  //an edge uid value that is free to use
  edge_id_type free_edge_;
  //performance statistics on this graph
  perf_stats_type stats_;

};

#endif
