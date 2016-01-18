#ifndef PARSER_H
#define PARSER_H

#include "imports.h"
#include "lexer.h"

// A parser is instantiated with an input stream (to read from), and a
// pointer to an interpreter instance.  parser::read_expr() can then
// be called repeatedly to read the next parsed expression from the
// stream; it returns NULL on EOF.

class parser {
  public:
    parser(istream &);
    value *read_expr();
  
  private:  
    value *read_list_rest();
    value *read_after_dot();
    lexer *_lex;
};

#endif // PARSER_H
