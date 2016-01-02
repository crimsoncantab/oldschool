#ifndef CS207_UTIL_HPP
#define CS207_UTIL_HPP

/**
 * @file Util.hpp
 * Common helper code for CS207
 *
 * @brief Some common utilities for use in CS207
 */

#include <iostream>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

namespace CS207 {

// Random number in [0,1)
inline double random() {
  return ::drand48();
}

// Random number in [A,B)
inline double random(double A, double B) {
  return A + (B - A) * random();
}

/**
 * Clock class, useful when timing code.
 */
class Clock {
 public:
  /** Construct a Clock and start timing. */
  Clock() {
    start();
  }
  /** Start the clock. */
  inline void start() {
    time_ = now();
  }
  /** Return the amount of time elapsed since the last start. */
  inline double elapsed() const {
    timeval tv = now();
    timersub(&tv, &time_, &tv);
    return tv.tv_sec + tv.tv_usec/1e6;
  }
 private:
  timeval time_;
  inline static timeval now() {
    timeval tv;
    gettimeofday(&tv, 0);
    return tv;
  }
};

/** Forces the thread to sleep for t seconds. Accurate to microseconds.
 */
inline int sleep(double t) {
  return usleep(t*1e6);
}

/** Read a line from @a s, parse it as type T, and store it in @a value.
 * @param[in]   s      input stream
 * @param[out]  value  value returned if the line in @a s doesn't parse
 *
 * If the line doesn't parse correctly, then @a s is set to the "failed"
 * state. Ignores blank lines and lines that start with '#'.
 */
template <typename T>
std::istream& getline_parsed(std::istream& s, T& value)
{
  // Get a line from the file
  std::string str;
  do {
    getline(s, str);
  } while (s && (str.empty() || str[0] == '#'));
  std::istringstream is(str);
  is >> value;
  if (is.fail())
    s.setstate(std::istream::failbit);
  return s;
}

} // end namespace CS207
#endif
