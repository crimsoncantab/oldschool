#include "value.h"
#include "exn.h"
#include "interp.h"

const char *value::tag_name(tag type)
{
    switch (type) {
        case PAIR:      return "pair";
        case PROCEDURE: return "procedure";
        case BOOLEAN:   return "boolean";
        case NUMBER:    return "number";
        case SYMBOL:    return "symbol";
        case NIL:       return "nil";
    }

    throw interp_bug("Unrecognized type: " + type);
}

void value::type_check(tag got, tag expected, string context)
{
    if (got != expected) {
        string msg;

        if ( ! context.empty() ) {
            msg += context + ": ";
        }

        msg += tag_name(got);
        msg += " where ";
        msg += tag_name(expected);
        msg += " expected";

        throw type_error(msg);
    }
}

string value::to_string() const
{
    ostringstream res;
    insert(res);
    return res.str();
}

ostream & operator<<(ostream & out, const value & v)
{
    v.insert(out);
    return out;
}

