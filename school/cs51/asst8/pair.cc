#include "value.h"

value::tag pair::ctype()
{
    return PAIR;
}

value::tag pair::type() const
{
    return PAIR;
}

pair::pair(value *argCar, value *argCdr) : car(argCar), cdr(argCdr) {}

void pair::insert(ostream & out) const
{
    out << "(";
    pair * temp = new pair(car, cdr);
    
    while (temp->type() == pair::ctype())
    {
        (temp->car)->insert(out);
        if ((temp->cdr)->type() == nil::ctype())
            break;
        if ((temp->cdr)->type() != pair::ctype())
        {
            out << " . ";
            (temp->cdr)->insert(out);
            break;
        }
        out << " ";
        temp = (pair *)temp->cdr;
    }

        out << ")";
}
