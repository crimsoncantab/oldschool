#include "value.h"

value::tag nil::ctype()
{
    return NIL;
}

value::tag nil::type() const
{
    return NIL;
}

nil *nil::instance()
{
    if (_instance == NULL) {
        _instance = new nil;
    }

    return _instance;
}

void nil::insert(ostream & out) const
{
    out << "()";
}

nil::nil() { }

nil *nil::_instance = NULL;

