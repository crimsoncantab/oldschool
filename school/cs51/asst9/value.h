#ifndef VALUE_H
#define VALUE_H

#include "imports.h"


#include "env.h"

// Abstract base class for values:
class value {
  public:
    //
    // TYPE STUFF
    //

    // Every value has exactly one of these six type tags:
    enum tag {
        NIL, BOOLEAN, PAIR, NUMBER, SYMBOL, PROCEDURE
    };

    // We'd like to get the name of a tag from the enum:
    static const char *tag_name(tag);

    // Assert that two type tags are equal, or throw an exception if
    // they aren't.
    static void type_check(tag got, tag expected, string context = "");

    // Safe casts -- throw a type exception if a value is of the wrong type.
    template <typename Out>
    Out *to(string context = "");

    template <typename Out>
    const Out *to(string context = "") const;

    virtual ~value() { }

    //
    // PURE VIRTUALS
    //

    // What is the type of this value?
    virtual tag type() const = 0;

    // Print the value on an ostream:
    virtual void insert(ostream &) const = 0;

    //
    // OPERATIONS: evaluation, printing
    //

    // Default eval() returns this; needs to be overridden for
    // some value sub-classes.
    virtual value *eval(env *locals);

    // If no locals, then we're in toplevel.  This just calls the virtual
    // eval above, anyway.
    value *eval();

    // Uses insert() to create a string.
    string to_string() const;
};

// nil is an example of the Singleton Pattern.  Since we want only one
// instance of nil to exist, ever, the constructor is private.  Get
// the instance of nil with nil::instance().
class nil : public value {
  public:
    static tag ctype();
    tag type() const;

    void insert(ostream &) const;
    static nil *instance();

  private:
    nil();
    static nil *_instance;
};

// boolean is a variation on the Singleton pattern -- call it a
// doubleton.  The instances are boolean::t() and boolean::f();
class boolean : public value {
  public:
    static tag ctype();
    tag type() const;

    static boolean *t();
    static boolean *f();
    static boolean *get(bool);

    void insert(ostream &) const;

    const bool prim;

  private:
    boolean(bool);

    static boolean *_t;
    static boolean *_f;
};

// Pairs are pretty straightforward.
class pair : public value {
  public:
    static tag ctype();
    tag type() const;

    pair(value *car, value *cdr);

    value *eval(env *locals);
    void insert(ostream &) const;

    value *car, *cdr;
};

class number : public value {
  public:
    static tag ctype();
    tag type() const;

    number(long);

    void insert(ostream &) const;

    const long prim;
};

// Symbols are interned, so we make the constructor private.
// symbol::intern(string) gets us a pointer to an interned
// symbol by its name, creating it if necessary.
class symbol : public value {
  public:
    static tag ctype();
    tag type() const;

    static symbol *intern(string);

    value *eval(env *locals);
    void insert(ostream &) const;

    const string name;

  private:
    symbol(const string);

};

// Procedures are apply-able function objects.  We'll sub-class
// procedure for user and primitive procs.
class procedure : public value {
  public:
    static tag ctype();
    tag type() const;

    procedure(const value *ltag);

    virtual value *operator()(value *args) = 0;

    const value *const ltag;    // procedure name
};

// User procedures take an env *locals so they can capture the
// environment.
class lambda : public procedure {
  public:
    lambda(const value *ltag, value *formals, value *body, env *locals);

    value *operator()(value *args);
    void insert(ostream &) const;

  private:
    env *const _locals;
    const value *_formals;
    const vector<value *> _body;
};

// Primitive procedures are instantiated from a function pointer of
// type func_t, which expects to receive the function arguments in a
// vector.
class primitive : public procedure {
  public:
    typedef value *(*func_t)(vector<value *>);

    primitive(func_t func, const value *ltag);

    value *operator()(value *args);
    void insert(ostream &) const;

  private:
    const func_t _func;
};

// The safe cast methods.
template <typename Out>
Out *value::to(string context)
{
    type_check(type(), Out::ctype(), context);
    return static_cast<Out *>(this);
}

template <typename Out>
const Out *value::to(string context) const
{
    type_check(type(), Out::ctype(), context);
    return static_cast<const Out *>(this);
}

// Values are insertable.  Use cout << *v, for example.
ostream & operator<<(ostream &, const value &);

#endif // VALUE_H
