#include "prims.h"
#include "util.h"
#include "interp.h"
#include "parser.h"

value *prim_apply(vector<value *> args)
{
    check_args(args, -2, "apply");

    procedure *proc = args[0]->to<procedure>("apply");
    value *nargs = args.back();

    for (int i = args.size() - 2; i > 0; --i) {
        nargs = cons(args[i], nargs);
    }

    return (*proc)(nargs);
}

value *prim_load(vector<value *> args)
{
    check_args(args, 1, "load");
    string filename = args[0]->to<symbol>("load")->name;
    return scheme.load(filename);
}

value *prim_exit(vector<value *> args)
{
    if (args.size() == 0) {
        exit(0);
    }

    if (args.size() > 1) {
        cerr << "exit: requires 0 or 1 arguments (giving up)\n";
    }

    try {
        exit(args[0]->to<number>("exit")->prim);
    } catch (type_error e) {
        cerr << e.msg() << endl;
        exit(1);
    }
}

value *prim_cons(vector<value *> args)
{
    check_args(args, 2, "cons");
    return cons(args[0], args[1]);
}

value *prim_car(vector<value *> args)
{
    check_args(args, 1, "car");
    return car(args[0], "car");
}

value *prim_cdr(vector<value *> args)
{
    check_args(args, 1, "cdr");
    return cdr(args[0], "cdr");
}


value *prim_eq(vector<value *> args)
{
    if (args.size() >= 2) {
        value *first = args[0];

        for (vector<value *>::iterator i = args.begin() + 1;
             i != args.end();
             ++i) {
            if (first != *i) {
                return boolean::f();
            }
        }
    }

    return boolean::t();
}

value *prim_num_eq(vector<value *> args) {
    if (args.size() >= 1) {
        number *first = args[0]->to<number>("=");

        for (vector<value *>::iterator i = args.begin() + 1;
             i != args.end();
             ++i) {
            if ( first->prim != (*i)->to<number>("=")->prim ) {
                return boolean::f();
            }
        }
    }

    return boolean::t();
}

value *prim_num_lt(vector<value *> args) {
    if (args.size() >= 1) {
        number *prev = args[0]->to<number>("<");

        for (vector<value *>::iterator i = args.begin() + 1;
             i != args.end();
             ++i) {
            if ( prev->prim < (*i)->to<number>("<")->prim ) {
                prev = (*i)->to<number>("<");
            } else {
                return boolean::f();
            }
        }
    }

    return boolean::t();
}

value *prim_num_le(vector<value *> args) {
    if (args.size() >= 1) {
        number *prev = args[0]->to<number>("<=");

        for (vector<value *>::iterator i = args.begin() + 1;
             i != args.end();
             ++i) {
            if ( prev->prim <= (*i)->to<number>("<=")->prim ) {
                prev = (*i)->to<number>("<=");
            } else {
                return boolean::f();
            }
        }
    }

    return boolean::t();
}

value *prim_display(vector<value *> args)
{
    for (vector<value *>::iterator i = args.begin();
         i != args.end();
         ++i) {
        cout << **i;
    }

    return nil::instance();
}

value *prim_newline(vector<value *> args)
{
    check_args(args, 0, "newline");
    cout << endl;

    return nil::instance();
}

value *prim_space(vector<value *> args)
{
    check_args(args, 0, "space");
    cout << ' ';

    return nil::instance();
}

value *prim_plus(vector<value *> args)
{
    long res = 0;

    for (vector<value *>::iterator i = args.begin(); i != args.end(); i++) {
        res += (*i)->to<number>("+")->prim;
    }

    return new number(res);
}

value *prim_times(vector<value *> args)
{
    long res = 1;

    for (vector<value *>::iterator i = args.begin(); i != args.end(); i++) {
        res *= (*i)->to<number>("*")->prim;
    }

    return new number(res);
}

value *prim_minus(vector<value *> args)
{
    if (args.size() == 0) {
        return new number(0);
    } else if (args.size() == 1) {
        return new number( - args[0]->to<number>("-")->prim );
    } else {
        long res = args[0]->to<number>("-")->prim;

        for (vector<value *>::iterator i = args.begin() + 1;
                i != args.end();
                i++) {
            res -= (*i)->to<number>("-")->prim;
        }

        return new number(res);
    }
}

value *prim_divides(vector<value *> args)
{
    if (args.size() == 0) {
        return new number(1);
    } else if (args.size() == 1) {
        return new number( 1 / args[0]->to<number>("-")->prim );
    } else {
        long res = args[0]->to<number>("/")->prim;

        for (vector<value *>::iterator i = args.begin() + 1;
                i != args.end();
                i++) {
            res /= (*i)->to<number>("/")->prim;
        }

        return new number(res);
    }
}

value *prim_modulo(vector<value *> args)
{
    check_args(args, 2, "%");

    return new number(args[0]->to<number>("%")->prim %
                      args[1]->to<number>("%")->prim);
}
