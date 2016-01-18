#include "value.h"

value::tag number::ctype()
{
    return NUMBER;
}

value::tag number::type() const
{
    return NUMBER;
}

number::number(long num) : prim(num) {}

void number::insert(ostream & out) const 
{
    cout << prim;
}
