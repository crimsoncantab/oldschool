#ifndef EVAL_H
#define EVAL_H

#include "value.h"

value *stx_quote        (value *, env *);
value *stx_if           (value *, env *);
value *stx_named_lambda (value *, env *);
value *stx_define       (value *, env *);
value *stx_lambda       (value *, env *);
value *stx_cond         (value *, env *);
value *stx_let          (value *, env *);

#endif // EVAL_H
