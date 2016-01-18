#include "util.h"
#include "exn.h"

pair *quote(value *v)
{
    return cons(symbol::intern("quote"),
                cons(v, nil::instance()));
}

pair *cons(value *car, value *cdr)
{
    return new pair(car, cdr);
}

value * & car(value *v, string context)
{
    return v->to<pair>(context)->car;
}

value * & cdr(value *v, string context)
{
    return v->to<pair>(context)->cdr;
}

const value *car(const value *v, string context)
{
    return v->to<pair>(context)->car;
}

const value *cdr(const value *v, string context)
{
    return v->to<pair>(context)->cdr;
}

vector<value *> get_args(value *actuals, int nargs, string name)
{
    vector<value *> res = list_to_vect(actuals);
    check_args(res, nargs, name);
    return res;
}

vector<value *> list_to_vect(value *actuals)
{
    vector<value *> res;

    while (actuals != nil::instance()) {
        res.push_back(car(actuals));
        actuals = cdr(actuals);
    }

    return res;
}

void check_args(const vector<value *> &args, int nargs, string name)
{
    if (nargs < 0) {
        if ((signed)args.size() < -nargs) {
            throw argument_error(name + ": requires at least " +
                                 stringify(-nargs) + " arguments");
        }
    } else if (nargs != (signed)args.size()) {
        throw argument_error(name + ": requires exactly " +
                             stringify(nargs) + " arguments");
    }
}

void check_proper_list(value * temp)
{
    while (1)
    {   if (cdr(temp)->type() != pair::ctype())
        {
            if (cdr(temp)->type() != nil::ctype())
                throw type_error("invalid lambda arguments");
            else
                break;
        } else
        temp = cdr(temp);
    }
}
