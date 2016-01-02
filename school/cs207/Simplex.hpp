#ifndef CS207_SIMPLEX_HPP
#define CS207_SIMPLEX_HPP
#include <iostream>

/** Class to represent and read Simplexes: tuples of N unsigned integers.
 */
template <int N>
struct Simplex {
  static constexpr int size = N;

  unsigned n[N];

  /** Read a Simplex from @a s.
   *
   * Simplexes are read as space-separated sets of integers. */
  friend std::istream& operator>>(std::istream& s, Simplex<N>& x) {
    for (int k = 0; k < N; ++k)
      s >> x.n[k];
    return s;
  }

  /** Write a Simplex to @a s. */
  friend std::ostream& operator>>(std::ostream& s, Simplex<N>& x) {
    for (int k = 0; k < N; ++k)
      s << x.n[k];
    return s;
  }
};

typedef Simplex<3> Triangle;
typedef Simplex<4> Tetrahedron;

#endif
