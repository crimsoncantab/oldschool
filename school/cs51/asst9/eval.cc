#include "value.h"
#include "util.h"
#include "interp.h"
#include "exn.h"
#include "eval.h"

// If no locals, we just eval in globals:
value *value::eval()
{
    return eval(&scheme.globals);
}

// All values evaluate to themselves unless they override this method.
value *value::eval(env *locals)
{
    return this;
}

//
// symbol::eval and pair::eval:
//

// YOUR CODE HERE  (our solution: less than 50 lines)
value *symbol::eval(env *locals)
{
    return ((*locals)[this])->eval(locals);
}

value *pair::eval(env *locals)
{
    value * newcar;
    if (car->type() == pair::ctype())
        newcar = car->eval(locals);
    else
        newcar = car;
    if (newcar->type() == symbol::ctype())
    {
        symbol * op = (symbol *) newcar;
        if (! (op->name).compare("quote") )
            return stx_quote(cdr, locals);
        else if (! (op->name).compare("if") )
            return stx_if(cdr, locals);
        else if (! (op->name).compare("named-lambda") )
            return stx_named_lambda(cdr, locals);
        else if (! (op->name).compare("lambda") )
            return stx_lambda(cdr, locals);
        else if (! (op->name).compare("define") )
            return stx_define(cdr, locals);
        else if (! (op->name).compare("let") )
            return stx_let(cdr, locals);
        else if (! (op->name).compare("cond") )
            return stx_cond(cdr, locals);
        else
            newcar = newcar->eval(locals); 
    }
    if (newcar->type() == primitive::ctype())
        return (*(newcar->to<primitive>()))(cdr->eval(locals));
    else if (newcar->type() == lambda::ctype())
        return (*(newcar->to<lambda>()))(cdr->eval(locals));
    else
        return cons(newcar, cdr->eval(locals));
}


// Quote just returns its argument unevaluated.
value *stx_quote(value *args, env *locals)
{
    return get_args(args, 1, "quote")[0];
}

value *stx_if(value *args, env *locals)
{
    vector<value *> v = list_to_vect(args);

    if (v.size() < 2 || v.size() > 3) {
        throw argument_error("if: requires 2 or 3 arguments");
    }

    if (v[0]->eval(locals) == boolean::f()) {
        if (v.size() == 3) {
            return v[2]->eval(locals);
        } else {
            return nil::instance();
        }
    } else {
        return v[1]->eval(locals);
    }
}

// Like lambda, but expects a "name" before the argument list, which it
// attaches to the new procedure.
value *stx_named_lambda(value *args, env *locals)
{
    vector<value *> v = get_args(args, -3, "named-lambda");
    value *ltag = v[0]->eval(locals);
    return new lambda(ltag, v[1], cdr(cdr(args)), locals);
}

// Delegate to named-lambda, with "-anon-" as the name.
value *stx_lambda(value *args, env *locals)
{
    vector<value *> v = get_args(args, -2, "lambda");

    // Build up a named-lambda expression:
    value *nl = cons(symbol::intern("named-lambda"),
                     cons(quote(symbol::intern("-anon-")),
                          cons(v[0], cdr(args))));

    // Recur into eval to delegate it to stx_named_lambda:
    value *result = nl->eval(locals);
    return result;
}

// Two cases:
//    (define name expr)
//    (define (name args ...) exprs ...)
// The latter becomes:
//    (define name (named-lambda name (args ...) exprs ...))
value *stx_define(value *args, env *locals)
{
    vector<value *> v = get_args(args, -2, "define");
    if (v[0]->type() == nil::ctype())
        throw type_error("define: nil where pair expected");
    if (v[0]->type() != pair::ctype())
    {
        v = get_args(args, 2, "define");
        value * tempv1 = v[1];
        locals->define((symbol *)v[0], tempv1);
        return tempv1->eval(locals);
    }
    else
    {
        value * goodcdr;
        if (cdr(v[0])->type() == pair::ctype())
            goodcdr = cdr(v[0]);
        else
            goodcdr = cons(cdr(v[0]), nil::instance());
        value * temp = cons(symbol::intern("named-lambda"),
                         cons(quote(car(v[0])),
                           cons(goodcdr, cdr(args))));
        locals->define(car(v[0])->to<symbol>(), temp);
        value *result = v[1];
        return result;
    }
}

//
// stx_cond and stx_let:
//

value *stx_cond(value *args, env *locals)
{
    if (car(args)->type() == nil::ctype())
        return nil::instance();
    value * condition = car(car(args))->eval(locals);
    
    if (condition == boolean::t() || condition->type() != boolean::ctype() )
        return cdr(car(args))->eval(locals);
    else
        return stx_cond(cdr(args), locals);
}

value *stx_let(value *args, env *locals)
{
    vector<value *> v = get_args(args, 2, "let");
    vector<value *> bindings = list_to_vect(v[0]);
    value * lambda_vars = nil::instance();
    value * lambda_vals = nil::instance();
    for (unsigned int i=0; i<bindings.size(); i++)
    {
        lambda_vars = cons(car(bindings[i]), lambda_vars);
        lambda_vals = cons(cdr(bindings[i]), lambda_vals);
    }
    value *newargs = cons(cons(symbol::intern("lambda"),
                            cons(lambda_vars, v[1])), lambda_vals);
    return newargs->eval(locals);
}
