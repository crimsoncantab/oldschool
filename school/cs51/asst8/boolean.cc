#include "value.h"

boolean *boolean::_t = NULL;
boolean *boolean::_f = NULL;

boolean::boolean(bool b) : prim(b) { }
 
value::tag boolean::ctype()
{
    return BOOLEAN;
}

value::tag boolean::type() const
{
    return BOOLEAN;
}

boolean *boolean::t()
{
    _t = new boolean(1);
    return _t;
}

boolean *boolean::f()
{
    _f = new boolean(0);
    return _f;
}

boolean *boolean::get(bool val)
{
    if (val)
        return t();
    else
        return f();
}

void boolean::insert(ostream & out) const
{
    
    out << "#" << (prim ? "t" : "f");
}
