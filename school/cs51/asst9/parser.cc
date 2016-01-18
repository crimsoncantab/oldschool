#include "parser.h"
#include "exn.h"

parser::parser(istream &is)
{
    _lex = new lexer(is);
}

value * parser::read_expr()
{   
    try{
        token first = _lex->get_next();
        if (first.type == token::LPAREN)
            return read_list_rest();
        else if (first.type == token::QUOTE)
            return new pair( symbol::intern("quote"), new pair(read_expr(),
              nil::instance()) );
        else if (first.type == token::VALUE)
            return first.get_value();
        else if (first.type == token::DOT)
        {   
            _lex->clear_line();
            throw syntax_error("dot unexpected", first.line, first.column);
        }
        else if (first.type == token::RPAREN)
        {   
            _lex->clear_line();
            throw syntax_error("')' unexpected", first.line, first.column);
        }
        else
            return NULL;
    }
    catch(syntax_error& error){
        throw error;
    }
}    
    


value * parser::read_list_rest()
{
    try{
        token first = _lex->get_next();
        if (first.type == token::RPAREN)
            return nil::instance();
        else
        {   _lex->push_back(first);
            value *expr1 = read_expr();
            token next = _lex->get_next();
            if (next.type == token::DOT)
                return new pair (expr1, read_after_dot());
            else
            {
                _lex->push_back(next);
                return new pair(expr1, read_list_rest());
                
            }
        }
    }
    catch(syntax_error& error){
        throw error;
    }
}

value * parser::read_after_dot()
{
    try{
        value *expr2 = read_expr();    
        token last = _lex->get_next();
        if (last.type != token::RPAREN)
        {   
            _lex->clear_line();
            throw syntax_error("')' expected", last.line, 
            last.column);
        }
        else
            return expr2;
    }
    catch(syntax_error& error){
        throw error;
    }

}
