// Forward declarations:
class interp;

#ifndef INTERPRETER_H
#define INTERPRETER_H

#include "imports.h"
#include "env.h"
#include "value.h"

extern interp & scheme;

class interp {
  public:
    void init();

    env globals;

    typedef map<const string, symbol *> symtab_t;
    symtab_t symtab;

    typedef value *(*syntax_t)(value *, env *);
    map<symbol *, syntax_t> syntax;

    value *load(string filename);

  private:
    void _bind_primitive(string, primitive::func_t);
    // YOUR CODE HERE  (our solution: less than 5 lines)
};

#endif // INTERPRETER_H
