// Forward declarations:
class env;

#ifndef ENV_H
#define ENV_H

#include "imports.h"
#include "value.h"

class symbol;
class value;

class env {
  public:
    env(env *next = NULL);

    value *operator[](const symbol *);
    void define(const symbol *, value *);

  private:
    // YOUR CODE HERE  (our solution: less than 5 lines)
    env * parent;
    map <const symbol *, value *> defs;
};

#endif // ENV_H
