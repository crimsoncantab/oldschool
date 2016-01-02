#ifndef CS207_ACTION_HPP
#define CS207_ACTION_HPP

#include "CS207/Util.hpp"

#include <SDL/SDL.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <string>
#include <assert.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

class Action {
public:
  SDLKey key;

  /**The iterators return nodes in graph m **/
  virtual void act(typename std::vector<unsigned>::iterator first,
    typename std::vector<unsigned>::iterator last) = 0;

};

//Sample actions

class Echoer : public Action {

  virtual void act(typename std::vector<unsigned>::iterator first,
    typename std::vector<unsigned>::iterator last) {
    for (; first != last; ++first) {
      std::cout << *first << " ";
    }
    std::cout << std::endl;
  }
  
};


#endif
