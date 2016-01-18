#include "imports.h"
#include "parser.h"
#include "value.h"
#include "exn.h"
#include "interp.h"

int main() try {
    parser mr_persnickety(cin);
    value *v;

    cout << "Welcome to Scheme51.  (Press Control-D to quit.)" << endl;

    scheme.init();

    for (;;) {
        try {
            cout << "> ";
            cout.flush();
            v = mr_persnickety.read_expr();

            if ( v == NULL ) {
                break;
            }

            v = v->eval();
            cout << *v << endl;
        } catch (fatal_error exn) {
            cerr << exn.msg() << endl;
            return 1;
        } catch (scheme_error exn) {
            cerr << exn.msg() << endl;
        }
    }

    return 0;
}

catch ( ... ) {
    cerr << "Uncaught exception . . . giving up" << endl;
    return 2;
}
