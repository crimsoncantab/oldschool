#ifndef MORTON_CODER_HPP
#define MORTON_CODER_HPP
#include "Point.hpp"
#include "BoundingBox.hpp"
#include <utility>
#include <assert.h>

// Automatically derive !=, <=, >, and >= from a class's == and <
using namespace std::rel_ops;

/** @file MortonCoder.hpp
 * @brief Define the MortonCoder class for Z-order-curve values, aka Morton
 *   codes.
 */

/** @class MortonCoder
 * @brief Class representing Z-order-curve values, aka Morton codes.
 *
 * The Z-order curve is a space-filling curve: a one-dimensional curve that
 * fills a multi-dimensional space. Space-filling curves offer advantages for
 * representing points in 3D space. Points near each other in 3D space are
 * likely to have close Morton codes! So we can store points in a map with
 * Z-order value as key, then iterate over nearby Z-order values to fetch
 * points nearby in space.
 *
 * Unfortunately, it's impossible to reduce 3D space to 1D space perfectly:
 * there are some discontinuities in any space-filling curve mapping. But the
 * mapping is still an awesome tool, and some optimizations (see BIGMIN on the
 * Wikipedia page linked below) make them very effective.
 *
 * The MortonCoder class encapsulates a BoundingBox and can be used to translate
 * between spatial Points and Morton codes relative to that BoundingBox.
 *
 * A single Morton code corresponds to a rectangular volume within that
 * BoundingBox, called its <em>cell</em>. Each side of the BoundingBox is
 * divided into 2^L equal-sized cells, for a total of 8^L cells.
 *
 * Read more about the Z-order curve here:
 *
 * http://en.wikipedia.org/wiki/Z-order_curve
 *
 * This class computes maps box numbers to point and visa-versa
 * with respect to a bounding box and the number of equal-volume boxes (8^L).
 * These mappings are performed in O(1) time.
 */
template <int L = 5 >
  class MortonCoder {
  // Using a 32-bit unsigned int for the code_type
  // means we can only resolve 10 3D levels
  static_assert(L >= 1 && L <= 10, "L (LEVELS) must be between 1 and 10");

public:

  typedef unsigned code_type;

  /** Type of this coder. */
  typedef MortonCoder<L> morton_type;

  /** The number of bits per dimension [octree subdivisions]. #cells = 8^L. */
  static constexpr int levels = L;
  /** The number of cells per side of the bounding box (2^L). */
  static constexpr code_type cells_per_side = code_type(1) << L;
  /** One more than the largest code (8^L). */
  static constexpr code_type end_code = code_type(1) << (3 * L);

  /** Construct a MortonCoder with a bounding box. */
  MortonCoder(const BoundingBox& bb)
    : pmin_(bb.min()),
    cell_size_((bb.max() - bb.min()) / cells_per_side) {
    assert(!bb.empty());
  }

  /** Return the MortonCoder's bounding box. */
  BoundingBox bounding_box() const {
    return BoundingBox(pmin_, pmin_ + (cell_size_ * cells_per_side));
  }

  /** Return the bounding box of the cell with Morton code @a c.
   * @pre c < end_code */
  BoundingBox cell(code_type c) const {
    assert(c < end_code);
    Point p = deinterleave(c);
    p *= cell_size_;
    p += pmin_;
    return BoundingBox(p, p + cell_size_);
  }

  /** Return the Morton code of Point @a p.
   * @pre bounding_box().contains(@a p)
   * @post cell(result).contains(@a p) */
  code_type code(const Point& p) const {
    Point s = (p - pmin_) / cell_size_;
    s.x = std::min(std::max(0.0, s.x), double(cells_per_side-1));
    s.y = std::min(std::max(0.0, s.y), double(cells_per_side-1));
    s.z = std::min(std::max(0.0, s.z), double(cells_per_side-1));
    return interleave((unsigned) s.x, (unsigned) s.y, (unsigned) s.z);
  }

  inline code_type bigmin(code_type c, code_type max, BoundingBox & b) const {
    max = std::min(max, end_code);
    const code_type m_x = 0x09249249;
    const code_type m_y = m_x << 1;
    const code_type m_z = m_x << 2;
    while (c < max && !(b & cell(c))) {
      code_type x = (c | (msb((c ^ max) & (m_x)) - 1)) + 1;
      code_type y = (c | (msb((c ^ max) & (m_y)) - 1)) + 1;
      code_type z = (c | (msb((c ^ max) & (m_z)) - 1)) + 1;
      c = std::max(x, std::max(y, z));
    }
    return c;
  }

  /** @class Morton::cell_iterator
   * @brief Iterator class for cells within a certain bounding box. */
  class cell_iterator {
  public:
    /** Element type. */
    typedef code_type value_type;
    /** Type of pointers to elements. */
    typedef code_type* pointer;
    /** Type of references to elements. */
    typedef code_type& reference;
    /** Iterator category. */
    typedef std::input_iterator_tag iterator_category;

    /** Construct an invalid cell_iterator. */
    cell_iterator() : morton_(), b_(), cell_(), max_() {
    }

    /** Return a cell that intersects the bounding box
     * @return the morton code of the current cell
     *
     * Complexity: O(1) */
    code_type operator*() const {
      return cell_;
    }

    /** Increment this iterator
     * @pre old code < max_code, where max_code is the smallest morton code
     * that is larger than that of any cell that intersect the bounding box.
     * @post for all i, old code < i < new code, the cell with code
     * i does not intersect the bounding box
     * @post new code is the code of a cell that intersects the bounding box,
     * or this iterator is past-the-end.
     * Complexity: O(1) */
    cell_iterator& operator++() {
      ++cell_;
      fix();
      return *this;
    }

    /** Test whether this iterator and @a x are equal.
     * @pre both iterators are for the same bounding box
     *
     * this iterator == @a x iff both refer to the same cell or both are
     * past-the-end.
     *
     * Complexity: O(1) */
    bool operator==(const cell_iterator& x) const {
      return (cell_ > max_ && x.cell_ > max_) || (cell_ == x.cell_);
    }

  private:
    friend class MortonCoder<L>;
    morton_type * morton_;
    BoundingBox b_;
    code_type cell_;
    code_type max_;

    cell_iterator(morton_type * const morton, const BoundingBox & b, const code_type & cell)
      : morton_(morton), b_(b), cell_(cell), max_(std::min(morton_->code(b.max()), morton_type::end_code)) {
      fix();
    }

    /** does BIGMIN*/
    void fix() {
      cell_ = morton_->bigmin(cell_, max_, b_);
    }
  };

  cell_iterator cell_begin(const BoundingBox& b) const {
    return cell_iterator(const_cast<morton_type *> (this), b & bounding_box(), code(b.min()));
  }

  cell_iterator cell_end(const BoundingBox& b) const {
    return cell_iterator(const_cast<morton_type *> (this), b & bounding_box(), std::min(code(b.max()), end_code) + 1);
  }

private:

  /** The minimum of the MortonCoder bounding box. */
  Point pmin_;
  /** The extent of a single cell. */
  Point cell_size_;

  /** Spreads the bits of a 10-bit number so that there are two 0s
   *  in between each bit.
   * @param x 10-bit integer
   * @return 28-bit integer of form 0b0000X00X00X00X00X00X00X00X00X00X,
   * where the X's are the original bits of @a x
   */
  inline unsigned spread_bits(unsigned x) const {
    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    x = (x | (x <<  8)) & 0b00000011000000001111000000001111;
    x = (x | (x <<  4)) & 0b00000011000011000011000011000011;
    x = (x | (x <<  2)) & 0b00001001001001001001001001001001;
    return x;
  }

  /** Interleave the bits of n into x, y, and z.
   * @pre x = [... x_2 x_1 x_0]
   * @pre y = [... y_2 y_1 y_0]
   * @pre z = [... z_2 z_1 z_0]
   * @post n = [... z_1 y_1 x_1 z_0 y_0 x_0]
   */
  inline code_type interleave(unsigned x, unsigned y, unsigned z) const {
    return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2);
  }

  /** Does the inverse of spread_bits, extracting a 10-bit number from
   * a 28-bit number.
   * @param x 28-bit integer of form 0bYYYYXYYXYYXYYXYYXYYXYYXYYXYYXYYX
   * @return 10-bit integer of form 0b00...000XXXXXXXXXX,
   * where the X's are every third bit of @a x
   */
  inline unsigned compact_bits(unsigned x) const {
    x &= 0b00001001001001001001001001001001;
    x = (x | (x >>  2)) & 0b00000011000011000011000011000011;
    x = (x | (x >>  4)) & 0b00000011000000001111000000001111;
    x = (x | (x >>  8)) & 0b00000011000000000000000011111111;
    x = (x | (x >> 16)) & 0b00000000000000000000001111111111;
    return x;
  }

  /** Deinterleave the bits from n into a Point.
   * @pre n = [... n_2 n_1 n_0]
   * @post result.x = [... n_6 n_3 n_0]
   * @post result.y = [... n_7 n_4 n_1]
   * @post result.z = [... n_8 n_5 n_2]
   */
  inline Point deinterleave(code_type c) const {
    return Point(compact_bits(c), compact_bits(c >> 1), compact_bits(c >> 2));
  }


  /** Smears the bits in c into the low bits by steps of three
   * including the original
   *
   * Example: 0000010000000000 -> 0000010010010010
   */
  inline unsigned smear_low_i3(unsigned c) const {
    c |= c >>  3;
    c |= c >>  6;
    c |= c >> 12;
    c |= c >> 24;
    return c;
  }
  
  /** @return @a c zeroed out except for the msb
   * Again, taken from:
   * http://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
   */
  inline code_type msb(code_type c) const {
    const unsigned int b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
    const unsigned int S[] = {1, 2, 4, 8, 16};
    int i;
    register unsigned int r = 0; // result of log2(v) will go here
    for (i = 4; i >= 0; i--) // unroll for speed...
    {
      if (c & b[i]) {
        c >>= S[i];
        r |= S[i];
      }
    }
    return r;
  }

  /** Smears the bits in c into the low bits by steps of one
   * excluding the high bit
   *
   * Example: 00011100100 -> 000011111111
   */
  inline unsigned smear_low_e1(unsigned c) const {
    c >>= 1;
    c |= c >>  1;
    c |= c >>  2;
    c |= c >>  4;
    c |= c >>  8;
    c |= c >> 16;
    return c;
  }
};

template <int L> constexpr typename MortonCoder<L>::code_type MortonCoder<L>::end_code;

#endif
