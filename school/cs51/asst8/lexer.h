#ifndef LEXER_H
#define LEXER_H

#include "imports.h"
#include "value.h"

// A token has a tag, which specifies what type it is.  token::VALUE
// tokens also have an associated value, which is retrieved using
// token::get_value().  If get_value() is called when the token type
// is not VALUE, it raises an interp_bug exception.
struct token {
  public:
    enum tag {
        LPAREN, RPAREN, DOT,
        QUOTE, END, VALUE,
    };

    token(int line, int column, tag t = END, value *val = NULL);
    value *get_value() const;
    const tag type;
    const int line, column;

  private:
    value *const _val;
};

// A lexer is instantiated with a stream.  It read the stream and
// breaks it up into tokens.  lexer::get_next() returns the next token
// from the stream, and lexer::push_back() pushes a token back onto
// the head of the stream.
class lexer {
  public:
    lexer(istream &);

    token get_next();
    void push_back(token);
    void clear_line(void);

  private:
    istream & _is;
    stack<token> _pushed;       // pushed back tokens
    bool _prompt;
    int line, column;

    // Abstract _is just a bit, keeping track of file location:
    int _get();
    void _push_back(char);
    void _eat_ws();

    // helper functions:
    token _read();              // read a token
    token _read_value(char);    // read the rest of a VALUE token

    // a simple buffer abstraction; should be self-explanatory.
    /*
    struct buf {
        char data[3192];
        size_t size;

        void add(char);
        void clear();
        void terminate();
        string str();
    } _buf;
    */
    string _buf;

    // get the token type associated with a particular character.
    static token::tag char_type(char);
};

#endif // LEXER_H
