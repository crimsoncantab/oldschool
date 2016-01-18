;;; tests.scm - tests for Scheme51

;;; This file contains tests grouped by what interpreter features they
;;; exercise.  You should uncomment the tests as you add the necessary
;;; features to your interpreter.  You may want to comment out basis.scm
;;; or the line (in C++) that loads it until your interpreter is ready
;;; to handle it.

;;; You can try this test file with our solution binary like this:
;;;
;;;     % scheme51.sol < test.scm
;;;
;;; and with your interpreter like this:
;;;
;;;     % make test

;;; There's a significant amount of code that you need to write before
;;; your interpreter will even compile.  One way to get around this is
;;; to figure out which functions you need to create and write stubs --
;;; for example:
;;;
;;;    value *do_lots_of_stuff(value *args, env *locals)
;;;    {
;;;        return NULL;  // TODO
;;;    }
;;;
;;; Of course, this works best if you can ensure that do_lots_of_stuff
;;; won't be invoked during that round of test.  Later, you can find
;;; where you wrote TODOs
;;;
;;;     % grep TODO *.h *.cc
;;;
;;; and go back and fill them in.

;;; Once you can get your interpreter to compile, even if you haven't
;;; added any of the features you need yet, it should be able to read
;;; and echo numbers, booleans, and nil:


;;; Each of these should be echoed with "> " in front of it:

 5
 89
 -4
 #f
 ()


;;; If you complete the implementation of class primitive, you should be
;;; able to echo primitives:

 <
 load
 modulo


;;; If you get function application working too, even with only stub
;;; environments, you should be able to call some primitives:

 (+ 1 2 3)
 (display 1) (display 2)


;;; Get special forms attached (even if you haven't written any yet),
;;; and then the four we've provided should work.

;;; These should echo without "quote":

 'foo
 '(bar baz qux)

;;; As should these:

 (if (eq? () ()) 'good 'bad)
 (if (eq? 4 5) 'bad 'good)


;;; Get symbol::intern right, and these should work:

 (eq? 'foo 'foo)                 ; => #t
 (eq? '(foo) '(foo))             ; => #f


;;; Once you have lambda::lambda and lambda::insert, you can create and
;;; print lambdas:

 (lambda (a b c) (+ a b c))
 (named-lambda 'identity (x) x)


;;; Get environments working, and write the first (name binding, not
;;; procedure definition) half of _define_.  Then bind some names:

 (define eight 8)
 (define true #t)
 (define hello 'hello)
 (define identity (named-lambda 'identity (x) x))


;;; With symbol::eval, you can also look up the bindings:

 eight
 true
 hello
 identity

;;; And with lambda::operator(), you can apply them:

 (identity '(3 5 7 11))
 ((lambda (x y z) (+ x (- y z))) 40 5 3)
 ((lambda nums (apply * nums)) 2 3 7)


;;; Finish up "define", and make sure lexical scope works:

 (define (f x y)
   ((lambda (z)
     (lambda (x) (cons x (cons y (cons z '())))))
    x))

 ((f 1 2) 3)             ; => (3 2 1)


;;; Write more tests for define, and tests for let and cond:
  (define (square x) (* x x))
  (square 3)              ; => 9
  
  (cond ((> 3 2) 'greater)
      ((< 3 2) 'less))    ; => greater
   
  (let ((x 10) (y 20))
            (+ x y))      ; => 30


;;; Here are some tests that should result in error messages:

 this-symbol-is-not-bound
 (+ 'cant-add 'symbols)

 (quote takes only one argument)

 (if takes two or three)
 (if not-one)

 (lambda (4 5 6) requires symbols for arguments)
 (lambda ((a)) not lists either)

 (define hmmmm 1 2 3)
 (define () 7)

 ((lambda (x) x) 'too 'many)
 ((lambda (x y z) x) 'too 'few)
