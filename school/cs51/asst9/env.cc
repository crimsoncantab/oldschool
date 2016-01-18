#include "env.h"
#include "exn.h"

// YOUR CODE HERE  (our solution: less than 20 lines)

env::env(env *next)
{
    parent = next;
}

void env::define(const symbol * sym, value * val)
{
    defs[sym] = val;
}

value * env::operator[](const symbol * lookup)
{
    if (defs[lookup])
        return defs[lookup];
    if (parent)
        return (*parent)[lookup];
    else
        throw not_bound(lookup->name);
}
