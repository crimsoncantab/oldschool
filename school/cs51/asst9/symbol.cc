#include "value.h"
#include "interp.h"


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
    if (scheme.symtab.find(str) == scheme.symtab.end())
        scheme.symtab[str] = new symbol(str);
    symbol * temp = scheme.symtab[str];
    return temp;
}
