#include "value.h"

value::tag symbol::ctype()
{
    return SYMBOL;
}

value::tag symbol::type() const
{
    return SYMBOL;
}

symbol::symbol(const string str) : name(str) {}

void symbol::insert(ostream & out) const
{
    out << name;
}

symbol *symbol::intern(string str)
{
    return new symbol(str);
}
