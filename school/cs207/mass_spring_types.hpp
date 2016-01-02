#ifndef MASS_SPRING_TYPES_HPP
#define	MASS_SPRING_TYPES_HPP

#include "Point.hpp"
#include "BoundingBox.hpp"
#include <algorithm>
//#define COLORS

struct node_state {
  double m_;
  Point v_;
  Point new_p_;
#ifdef COLORS
  enum visit_status {
    MISSED = 0, SKIPPED = 1, BOXED = 2, CONSTRAINED = 3
  } visit_status_;
  inline void update_status(visit_status status) const {
    const_cast<node_state *> (this)->visit_status_ = std::max(visit_status_, status);
  }
#endif

};

struct edge_state {
  double k_;
  double rest_l_;
  bool calced_;
  Point force_;
};

struct triangle_state {
  Point force_;
};

const BoundingBox WORLD = BoundingBox(Point(-5.0, -5.0, -5.0), Point(5.0, 5.0, 5.0));
//const BoundingBox WORLD = BoundingBox(Point(-50.0, -50.0, -50.0), Point(50.0, 50.0, 50.0));

#endif	/* MASS_SPRING_TYPES_HPP */

