#ifndef CS207_SPATIALGRAPH_HPP
#define CS207_SPATIALGRAPH_HPP

/**
 * @file Graph.hpp
 */

#include "CS207/Util.hpp"
#include "Point.hpp"
#include "Graph.hpp"
#include "BoundingBox.hpp"
#include "MortonCoder.hpp"
#include "mass_spring_types.hpp"
#include <utility>
#include <algorithm>
#include <vector>
#include <set>
#include <assert.h>

// Automatically derive !=, <=, >, and >= from a class's == and <
using namespace std::rel_ops;

/** @class SpatialGraph
 * @brief A template for 3D undirected graphs.
 *
 * Identical to a Graph except it provides spatial information.
 * All operations have the same complexity and interface as Graph
 * unless otherwise noted. */
template <typename N, typename E, int L = 5 >
  class SpatialGraph {
  typedef MortonCoder<L> morton_type;

  struct node_info {
    N value_;
    Point p_;
  };

  typedef Graph<node_info, E> graph_type;

  typedef std::pair<typename morton_type::code_type, typename graph_type::Node> morton_pair;

  typedef std::set<morton_pair> spatial_set_type;


public:

  // PUBLIC TYPE DEFINITIONS

  struct PerfStats {
    //total number of nodes examined by all neighborhood_iterator's
    unsigned neigh_total_;
    //total number of nodes examined and skipped by all neighborhood_iterator's
    unsigned neigh_skipped_;

    PerfStats() : neigh_total_(0), neigh_skipped_(0) {

    }
  };
  typedef PerfStats perf_stats_type;

  /** Type of this graph. */
  typedef SpatialGraph<N, E, L> spatial_graph_type;

  class Node;
  typedef Node node_type;

  class Edge;
  typedef Edge edge_type;

  typedef N node_value_type;
  typedef E edge_value_type;


  typedef unsigned size_type;

  template <typename IT, typename VAL>
  class base_iterator;

  typedef base_iterator<typename graph_type::node_iterator, Node> node_iterator;

  typedef base_iterator<typename graph_type::edge_iterator, Edge> edge_iterator;

  typedef base_iterator<typename graph_type::incident_iterator, Edge> incident_iterator;

  // CONSTRUCTOR AND DESTRUCTOR

  SpatialGraph(const BoundingBox & bounding_box = WORLD) :
    graph_(), morton_(bounding_box), spatial_nodes_(), stats_() {
  }

  ~SpatialGraph() {
  }


  // NODES

  class Node {
  public:

    Node() : sg_(NULL), graph_node_() {
    }

    spatial_graph_type* graph() const {
      return sg_;
    }

    Point position() const {
      return graph_node_.value().p_;
    }

    /** Set this node's position.
     * Complexity: O(log(size()))
     */
    void set_position(const Point & p) {
      assert(sg_->morton_.bounding_box().contains(p));
      sg_->spatial_update_node(graph_node_, graph_node_.value().p_, p);
      graph_node_.value().p_ = p;
    }

    const node_value_type & value() const {
      return graph_node_.value().value_;
    }

    node_value_type & value() {
      return graph_node_.value().value_;
    }

    size_type index() const {
      return graph_node_.index();
    }

    bool operator==(const Node& x) const {
      return (sg_ == x.sg_) && (index() == x.index());
    }

    bool operator<(const Node& x) const {
      return (sg_ < x.sg_) || (sg_ == x.sg_ && index() < x.index());
    }

    size_t degree() const {
      return graph_node_.degree();
    }

    incident_iterator edge_begin() const {
      return incident_iterator(sg_, graph_node_.edge_begin());
    }

    incident_iterator edge_end() const {
      return incident_iterator(sg_, graph_node_.edge_end());
    }

  private:
    friend class SpatialGraph<N, E, L>;
    spatial_graph_type* sg_;
    typename graph_type::Node graph_node_;

    Node(spatial_graph_type * const sg, const typename graph_type::Node & graph_node) : sg_(sg), graph_node_(graph_node) {
    }
  };

  const PerfStats & stats() const {
    return stats_;
  }

  void reset_stats() {
    stats_ = perf_stats_type();
  }

  size_type size() const {
    return graph_.size();
  }

  size_type num_nodes() const {
    return graph_.size();
  }

  Node node(size_type i) const {
    return Node(const_cast<spatial_graph_type *> (this), graph_.node(i));
  }

  /** Add a node to the graph, returning the added node.
   * @pre @a position is within the graph's world
   * @param[in] position The new node's position
   * Complexity: O(log(size())) amortized time. */
  Node add_node(const Point& position,
    const node_value_type& value = node_value_type()) {
    assert(morton_.bounding_box().contains(position));
    node_info info;
    info.p_ = position;
    info.value_ = value;
    typename graph_type::Node n = graph_.add_node(info);

    spatial_add_node(n, position);

    return Node(const_cast<spatial_graph_type *> (this), n);
  }

  //  TODO this doesn't work correctly with node indices...

  void remove_node(Node n) {
    spatial_remove_node(n.graph_node_, n.position());
    graph_.remove_node(n.graph_node_);
  }

  void clear() {
    graph_.clear();
    spatial_nodes_.clear();
    stats_ = perf_stats_type();

    assert(graph_.size() == 0);
  }

  // EDGES

  class Edge {
  public:

    Edge() : sg_(NULL), graph_edge_() {
    }

    Node node1() const {
      return Node(sg_, graph_edge_.node1());
    }

    Node node2() const {
      return Node(sg_, graph_edge_.node2());
    }

    bool operator==(const Edge& x) const {
      return x.graph_edge_ == graph_edge_;
    }

    const edge_value_type & value() const {
      return graph_edge_.value();
    }

    edge_value_type & value() {
      return graph_edge_.value();
    }

    /** The length of this edge
     * @pre this is a valid edge.
     * @return the distance between node1() and node2()
     *
     */
    double length() const {
      Point & p1 = graph_edge_.node1().value().p_;
      Point & p2 = graph_edge_.node2().value().p_;
      return (p1 - p2).length();
    }
  private:
    friend class SpatialGraph<N, E, L>;
    spatial_graph_type * sg_;
    typename graph_type::Edge graph_edge_;

    Edge(spatial_graph_type * const g, const typename graph_type::Edge & graph_edge)
      : sg_(g), graph_edge_(graph_edge) {
    }
  };

  size_type num_edges() const {
    return graph_.num_edges();
  }

  Edge edge(size_type i) const {
    return Edge(const_cast<spatial_graph_type *> (this), graph_.edge(i));
  }

  bool has_edge(const Node& a, const Node& b) const {
    return graph_.has_edge(a.graph_node_, b.graph_node_);
  }

  Edge add_edge(const Node& a, const Node& b, const edge_value_type& value = edge_value_type()) {
    return Edge(const_cast<spatial_graph_type *> (this), graph_.add_edge(a.graph_node_, b.graph_node_, value));
    ;
  }

  size_type remove_edge(const Node& a, const Node& b) {
    return graph_.remove_edge(a.graph_node_, b.graph_node_);
  }

  void remove_edge(const Edge& e) {
    graph_.remove_edge(e.graph_edge_);
  }


  // ITERATORS

  /** @class SpatialGraph::base_iterator
   * @brief Iterator template for node/edge iterators. A forward iterator. */
  template <typename IT, typename V>
  class base_iterator {
  public:
    typedef V value_type;
    typedef V* pointer;
    typedef V& reference;
    typedef std::input_iterator_tag iterator_category;

    base_iterator() : sg_(NULL), it_() {
    }

    V operator*() const {
      return V(sg_, *it_);
    }

    base_iterator& operator++() {
      ++it_;
      return *this;
    }

    bool operator==(const base_iterator<IT, V>& x) const {
      return it_ == x.it_;
    }

    bool operator<(const base_iterator<IT, V>& x) const {
      return it_ < x.it_;
    }

  private:
    friend class SpatialGraph<N, E, L>;
    spatial_graph_type * sg_;
    IT it_;

    base_iterator(spatial_graph_type * const sg, const IT it) : sg_(sg), it_(it) {
    }

  };

  node_iterator node_begin() const {

    return node_iterator(const_cast<spatial_graph_type *> (this), graph_.node_begin());
  }

  node_iterator node_end() const {
    return node_iterator(const_cast<spatial_graph_type *> (this), graph_.node_end());
  }

  edge_iterator edge_begin() const {
    return edge_iterator(const_cast<spatial_graph_type *> (this), graph_.edge_begin());
  }

  edge_iterator edge_end() const {
    return edge_iterator(const_cast<spatial_graph_type *> (this), graph_.edge_end());
  }

  /** @class Graph::neighboorhood_iterator
   * @brief Iterator class for nodes within a certain bounding box. */
  class neighborhood_iterator {
  public:

    typedef Node value_type;
    typedef Node* pointer;
    typedef Node& reference;
    typedef std::input_iterator_tag iterator_category;

    /** Construct an invalid neighboorhood_iterator. */
    neighborhood_iterator() : g_(NULL), b_(), map_(), it_(), max_() {
    }

    /** Return a node within this neighborhood
     * @return the current Node
     *
     * Complexity: O(1) */
    Node operator*() const {
      return Node(g_, (*it_).second);
    }

    /** Increment this iterator
     * @post if old *iterator refers to a node with morton code x, new *iterator
     * refers to a different node within b with a morton code >= x.  If no node satisfies
     * this requirement, new iterator is past-the-end.  For a given iterator,
     * this function will never point the iterator to the same node more than
     * once.
     *
     * Complexity: O(1) */
    neighborhood_iterator& operator++() {
//      typename morton_type::code_type old_code = (*it_).first;
      ++it_;
//      fix(old_code);
      fix();
      return *this;
    }

    /** Test whether this iterator and @a x are equal.
     * @pre both iterators are for the same graph
     *
     * this iterator == @a x iff both refer to the same node or both are
     * past-the-end.
     *
     * Complexity: O(1) */
    bool operator==(const neighborhood_iterator& x) const {
      return (it_ == x.it_);
    }

  private:
    friend class SpatialGraph<N, E, L>;

    spatial_graph_type * g_;
    BoundingBox b_;
    const spatial_set_type & map_;
    typename spatial_set_type::const_iterator it_;
    typename morton_type::code_type max_;

    neighborhood_iterator(spatial_graph_type * g, const BoundingBox& b, const spatial_set_type & map,
      typename spatial_set_type::const_iterator it, typename morton_type::code_type max)
      : g_(g), b_(b), map_(map), it_(it), max_(max) {
//      fix(0);
      fix();
    }

    void fix() {
//    void fix(typename morton_type::code_type old_cell) {
//      int i = 0;
      while (it_ != map_.end() && (*it_).first <= max_ && !b_.contains((*it_).second.value().p_)) {
//        if ((*it_).first != old_cell) {
//          //we're in a new cell
//          typename morton_type::code_type new_cell = g_->morton_.bigmin((*it_).first, max_, b_);
//          if ((*it_).first != new_cell) {
//            //the new cell did not intersect the bounding box
//            it_ = map_.lower_bound(std::make_pair(new_cell, g_->graph_.node(0)));
//          } else {
//            //the new cell does intersect the bounding box
//            //but the current node is not in the bounding box
#ifdef COLORS
            (*it_).second.value().value_.update_status(node_state::SKIPPED);
#endif
//            ++i;
            ++it_;
//          }
//          //update to the new cell
//          old_cell = new_cell;
//        } else {
//          //we're in the same cell
//          //but the current node is not in the bounding box
//#ifdef COLORS
//          (*it_).second.value().update_status(node_state::SKIPPED);
//#endif
//          ++i;
//          ++it_;
//        }
//      }
//#ifdef COLORS
//      if (((*it_).first <= max_)) {
//        (*it_).second.value().update_status(node_state::BOXED);
      }
//#endif
//      g_->stats_.neigh_total_ += i + 1;
//      g_->stats_.neigh_skipped_ += i;
    }

  };

  neighborhood_iterator node_begin(const BoundingBox& b) const {
    return neighborhood_iterator(const_cast<spatial_graph_type *> (this), b, spatial_nodes_,
      spatial_nodes_.lower_bound(std::make_pair(morton_.code(b.min()), graph_.node(0))), morton_.code(b.max()));
  }

  neighborhood_iterator node_end(const BoundingBox& b) const {
    return neighborhood_iterator(const_cast<spatial_graph_type *> (this), b, spatial_nodes_,
      spatial_nodes_.upper_bound(std::make_pair(morton_.code(b.max())+1, graph_.node(0))), morton_.code(b.max()));
  }

private:

  void spatial_add_node(const typename graph_type::Node n, const Point & position) {
    spatial_nodes_.insert(std::make_pair(morton_.code(position), n));
  }

  void spatial_remove_node(const typename graph_type::Node n, const Point & position) {
    spatial_nodes_.erase(spatial_nodes_.find(std::make_pair(morton_.code(position), n)));
  }

  void spatial_update_node(const typename graph_type::Node n, const Point & position_old, const Point & position_new) {
    typename morton_type::code_type oldc = morton_.code(position_old), newc = morton_.code(position_new);
    if (oldc != newc) {
      spatial_nodes_.erase(spatial_nodes_.find(std::make_pair(oldc, n)));
      spatial_nodes_.insert(std::make_pair(newc, n));
    }
  }

  graph_type graph_;
  //used to order nodes spacially
  morton_type morton_;
  //spatial ordering of nodes, based on morton_ ordering
  spatial_set_type spatial_nodes_;
  //performance statistics on this spatial graph
  perf_stats_type stats_;

};

#endif
