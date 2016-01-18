#include "interp.h"
#include "prims.h"
#include "parser.h"
#include "value.h"
#include "eval.h"

interp & scheme = *(new interp);

void interp::init()
{
    // YOUR CODE HERE  (our solution: less than 10 lines)

    _bind_primitive("apply", prim_apply);
    _bind_primitive("load", prim_load);
    _bind_primitive("exit", prim_exit);

    _bind_primitive("cons", prim_cons);
    _bind_primitive("cdr", prim_cdr);
    _bind_primitive("car", prim_car);

    _bind_primitive("null?", prim_type_pred<nil>);
    _bind_primitive("pair?", prim_type_pred<pair>);
    _bind_primitive("number?", prim_type_pred<number>);
    _bind_primitive("symbol?", prim_type_pred<symbol>);
    _bind_primitive("boolean?", prim_type_pred<boolean>);
    _bind_primitive("procedure?", prim_type_pred<procedure>);

    _bind_primitive("eq?", prim_eq);

    _bind_primitive("=", prim_num_eq);
    _bind_primitive("<", prim_num_lt);
    _bind_primitive("<=", prim_num_le);
    _bind_primitive("+", prim_plus);
    _bind_primitive("*", prim_times);
    _bind_primitive("-", prim_minus);
    _bind_primitive("/", prim_divides);
    _bind_primitive("modulo", prim_modulo);

    _bind_primitive("display", prim_display);
    _bind_primitive("newline", prim_newline);
    _bind_primitive("space", prim_space);

    load("basis.scm");
}

value *interp::load(string filename)
{
    ifstream ifs(filename.c_str());

    if ( ifs.good() ) {
        parser parsizzle(ifs);
        vector<value *> exprs;

        try {
            while ( value *val = parsizzle.read_expr() ) {
                exprs.push_back(val);
            }
        } catch (read_error exn) {
            cerr << exn.msg() << endl;
            cerr << "Warning: " << filename << ": read error (giving up)\n";
            return boolean::f();
        }

        for (vector<value *>::iterator i = exprs.begin();
                i != exprs.end();
                ++i) {
            try {
                (*i)->eval();
            } catch (scheme_error exn) {
                cerr << exn.msg() << endl;
            }
        }
    } else {
        cerr << "Warning: " << filename << ": could not open\n";
        return boolean::f();
    }

    return boolean::t();
}

void interp::_bind_primitive(string name, primitive::func_t func)
{
    symbol *sym = symbol::intern(name);
    globals.define(sym, new primitive(func, sym));
}

// YOUR CODE HERE  (our solution: less than 10 lines)
