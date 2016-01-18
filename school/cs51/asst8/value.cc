#include "value.h"
#include "exn.h"
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

