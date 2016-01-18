#include "value.h"

value::tag procedure::ctype()
{
    return PROCEDURE;
}

value::tag procedure::type() const
{
    return PROCEDURE;
}

procedure::procedure(const value *argLtag) : ltag(argLtag) {}

