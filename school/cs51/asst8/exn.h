#ifndef EXN_H
#define EXN_H

#include "imports.h"

class scheme_error {
  public:
    scheme_error(string msg = "Scheme error") : _msg(msg) { }
    virtual ~scheme_error() { }

    virtual const string & msg() const {
        return _msg;
    }

    template <typename T>
    static string stringify(const T & x) {
        ostringstream o;
        o << x;
        return o.str();
    }

  private:
    const string _msg;
};

class fatal_error : public scheme_error {
  public:
    fatal_error(string desc = "Fatal error") :
        scheme_error(desc)
    { }
};

class interp_bug : public fatal_error {
  public:
    interp_bug(string desc) :
        fatal_error("Fatal interpreter bug: " + desc)
    { }
};

class read_error : public scheme_error {
  public:
    read_error(string name, int line, int column) :
        scheme_error(name + " at " + stringify(line) +
                     ":" + stringify(column))
    { }
};

class lexical_error : public read_error {
  public:
    lexical_error(string name, int line, int column) :
        read_error("Lexical error: " + name, line, column)
    { }
};

class syntax_error : public read_error {
  public:
    syntax_error(string name, int line, int column) :
        read_error("Syntax error: " + name, line, column)
    { }
};

class runtime_error : public scheme_error {
  public:
    runtime_error(string msg = "Runtime error") :
        scheme_error(msg)
    { }
};

class not_bound : public runtime_error {
  public:
    not_bound(string name) :
        runtime_error("Name not bound: " + name)
    { }
};

class type_error : public runtime_error {
  public:
    type_error(string msg) :
        runtime_error("Type error: " + msg)
    { }
};

class argument_error : public runtime_error {
  public:
    argument_error(string msg) :
        runtime_error("Argument error: " + msg)
    { }
};

#endif // EXN_H
