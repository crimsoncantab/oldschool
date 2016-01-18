#ifndef PRIMS_H
#define PRIMS_H

#include "value.h"
#include "exn.h"

value *prim_apply(vector<value *>);
value *prim_load(vector<value *>);
value *prim_exit(vector<value *>);

value *prim_cons(vector<value *>);
value *prim_car(vector<value *>);
value *prim_cdr(vector<value *>);

value *prim_eq(vector<value *>);

value *prim_num_eq(vector<value *>);
value *prim_num_lt(vector<value *>);
value *prim_num_le(vector<value *>);
value *prim_plus(vector<value *>);
value *prim_times(vector<value *>);
value *prim_minus(vector<value *>);
value *prim_divides(vector<value *>);
value *prim_modulo(vector<value *>);

value *prim_display(vector<value *>);
value *prim_newline(vector<value *>);
value *prim_space(vector<value *>);

template <typename T>
value *prim_type_pred(vector<value *> args)
{
    if (args.size() != 1) {
        string name = (T::ctype() == value::NIL)
                        ? "null"
                        : value::tag_name(T::ctype());
        throw scheme_error(name + "?: requires 1 argument");
    }

    return boolean::get(args[0]->type() == T::ctype());
}

#endif // PRIMS_H
