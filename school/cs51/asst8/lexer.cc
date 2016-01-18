#include <cctype>
#include <cstdlib>
#include "lexer.h"
#include "exn.h"

//
// CLASS TOKEN
//

token::token(int line, int column, tag type, value *val) :
    type(type), line(line), column(column), _val(val)
{ }

value *token::get_value() const
{
    if (type == VALUE) {
        return _val;
    } else {
        throw interp_bug("attempt to get value from non-value token");
    }
}

//
// CLASS LEXER
//

lexer::lexer(istream & is) : _is(is) {
    line = 1;
    column = 0;
}

token lexer::get_next()
{
    if (_pushed.size() > 0) {
        token res = _pushed.top();
        _pushed.pop();
        return res;
    }

    return _read();
}

token lexer::_read()
{
    for (_eat_ws(); _is.good(); _eat_ws()) {
        char c = _get();

        if (c == ';') {
            while (c != '\n' && _is.good()) {
                c = _get();
            }
        } else {
            token::tag type = lexer::char_type(c);

            if (type == token::VALUE) {
                return _read_value(c);
            } else {
                return token(line, column, type);
            }
        }
        // loop for comments
    }

    return token(line, column, token::END);
}

int lexer::_get()
{
    int c = _is.get();

    if (c == '\n') {
        ++line;
        column = 0;
    } else {
        ++column;
    }

    return c;
}

void lexer::_push_back(char c)
{
    if (c == '\n') {
        --line;
        column = -1;
    } else {
        --column;
    }
    _is.putback(c);
}

void lexer::_eat_ws()
{
    char c = _get();

    while (isspace(c)) {
        c = _get();
    }

    _push_back(c);
}

token lexer::_read_value(char c)
{
    value *val;

    _buf.clear();

    for ( ; _is.good(); c = _get() ) {
        if (isspace(c)) {
            break;
        } else if (lexer::char_type(c) == token::VALUE) {
            _buf.push_back(tolower(c));
        } else {
            _push_back(c);
            break;
        }
    }

    if (_buf.size() == 2 && _buf[0] == '#') {
        if (_buf[1] == 't') {
            val = boolean::t();
        } else if (_buf[1] == 'f') {
            val = boolean::f();
        } else {
            throw lexical_error(_buf, line, column);
        }
    } else {
        char *endptr = NULL;
        const char *beginptr = _buf.c_str();
        long num = strtol(beginptr, &endptr, 0);

        if (endptr - beginptr == (ssize_t)_buf.size()) {
            val = new number(num);
        } else {
            val = symbol::intern(_buf);
        }
    }

    return token(line, column, token::VALUE, val);
}

void lexer::push_back(token tok)
{
    _pushed.push(tok);
}

void lexer::clear_line()
{
    char c = _get();

    while (c != '\n' && _is.good()) {
        c = _get();
    }

    while ( ! _pushed.empty() ) {
        _pushed.pop();
    }
}

token::tag lexer::char_type(char c)
{
    switch (c) {
      case '(': case '[':
        return token::LPAREN;

      case ')': case ']':
        return token::RPAREN;

      case '.':
        return token::DOT;

      case '\'':
        return token::QUOTE;

      default:
        return token::VALUE;
    }
}
