#ifndef UTIL_H
#define UTIL_H

#include "imports.h"
#include "value.h"

// A few shortcuts:
pair *quote(value *v);
pair *cons(value *car, value *cdr);
value * & car(value *p, string = "");
value * & cdr(value *p, string = "");
const value *car(const value *p, string = "");
const value *cdr(const value *p, string = "");

//Check for a proper list
void check_proper_list(value *temp);


// Argument checking:

//   Convert a Scheme list to an STL vector:
vector<value *> list_to_vect(value *actuals);

//   If n is non-negative, check for n args; if n is negative, check for
//   >= abs(n) args.  If not, raise an exception with name for context.
void check_args(const vector<value *> &args, int n, string name);

//   Do both list_to_vect and check_args:
vector<value *> get_args(value *actuals, int n, string name);

template <typename T>
string stringify(const T & x) {
    ostringstream o;
    o << x;
    return o.str();
}

#endif // UTIL_H
