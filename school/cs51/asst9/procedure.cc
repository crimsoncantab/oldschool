#include "value.h"
#include "util.h"
#include "exn.h"
#include "eval.h"

value::tag procedure::ctype()
{
    return PROCEDURE;
}

value::tag procedure::type() const
{
    return PROCEDURE;
}

procedure::procedure(const value *argLtag) : ltag(argLtag) {}


primitive::primitive(func_t func, const value *ltag) : 
procedure(ltag), _func(func) {}

value * primitive::operator()(value *args)
{
    return _func(list_to_vect(args));
}

void primitive::insert(ostream &out) const
{
    out << "#<primproc:";
    ltag->insert(out);
    out << ">";
}


void lambda::insert(ostream &out) const
{
    out << "#<userproc:";
    ltag->insert(out);
    out << ">";
}

lambda::lambda(const value *ltag, value *formals, value *body, env *locals) :
procedure(ltag), _locals(locals), _formals(formals), _body(list_to_vect(body))
{
    // make sure formals is a single symbol or list of symbols
    value *temp = formals;
    while (1)
    {
        if (temp->type() == nil::ctype())
            break;
        if (temp->type() != pair::ctype()) {
            if (temp->type() != symbol::ctype())
                throw type_error("invalid lambda arguments");
            else
                break;
        }
        else if (car(temp)->type() != symbol::ctype())
            throw type_error("invalid lambda arguments");
        
        temp = cdr(temp);
    }
    // make sure body is proper list
    check_proper_list(body);
}

value * lambda::operator()(value *args)
{
    check_proper_list(args);
    env * newscope = new env(_locals);
    value * newformals;
    if (_formals->type() == symbol::ctype())
        newformals = cons(const_cast<value *>(_formals), nil::instance());
    else
        newformals = const_cast<value *>(_formals);
    vector<value *> formal_vec = list_to_vect(newformals);
    get_args(args, -formal_vec.size(), "lambda");
    value * temp = args;
    
    for (unsigned int i=0; i<formal_vec.size(); i++)
    {
        if (i != formal_vec.size()-1)
        {
            newscope->define((formal_vec[i])->to<symbol>(), car(temp));
            temp = cdr(temp);
        }
        else
        {
            if (cdr(temp)->type() == nil::ctype())
                temp = car(temp);
            newscope->define((formal_vec[i])->to<symbol>(), temp);
        }
    }
    // evaluate each expression of body
    for (unsigned int i=0; i<_body.size()-1; i++)
         _body[i]->eval(newscope);
    return _body[_body.size()-1]->eval(newscope);
}
