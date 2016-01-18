#ifndef VALUE_H
#define VALUE_H

#include "imports.h"
#include "exn.h" //needed for the downCast function
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


// Values are insertable.  Use cout << *v, for example.
ostream & operator<<(ostream &, const value &);

template <class castType>
castType * downCast(value *val) {
    if(val->type() != castType::ctype()) {
	throw type_error("Attempting to down-cast non-matching type");
    }
    return dynamic_cast<castType *>(val);

}

#endif // VALUE_H
