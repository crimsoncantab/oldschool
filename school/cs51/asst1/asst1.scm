;; CS51 Assignment 1
;; 
;; File:          asst1.scm
;; Author:        Loren McGinnis <mcginn@fas.harvard.edu>
;; Collaborators: None
;; 
;; Collaboration is not permitted on this assignment.
;; By submitting this assignment, you certify that you have
;; followed the CS51 collaboration policy.  If you are unclear
;; on the CS51 collaboration policy, please ask your TF or
;; the head TF.


;;;;;
;;;;; Exercise 1
;;;;;

;; Each expression typed into the Scheme REPL contains a blank
;; (_).  What should go in the blanks to produce the results shown?

;;; (a)
;
;; > (cons 'foo _)
;; (foo)
;
; '()

;;; (b)
;
;; > (append _ '())
;; (bar)
;
; 'bar

;;; (c)
;
;; > (car (_ (cdr '(a (b c) d))))
;; b
;
; car


;;;;;
;;;;; Exercise 2
;;;;;


;;; (a)
;
;; What will this function return?  If it generates an error, explain
;; why.
;; > (cons 7 (list 5 6 7 8))
;
; => (7 5 6 7 8)

;;; (b)
;
;; What will this function return?  If it generates an error, explain
;; why.
;; > (append 7 (list 5 6 7 8))
;
; Returns an error, because 7 is not a list and append expects two lists.

;;; (c)
;
;; Rewrite the second function, still using append, so that it returns
;; the same value as the first function.
;
; (append (list 7) (list 5 6 7 8))


;;;;;
;;;;; Exercise 3
;;;;;

;; What does mystery do?  Describe it in clear English.
;; For example, suppose you were given this function: (define
;; (example x) (car (cdr x)))
;; 
;; We'd favor an answer such as, ``this function takes a
;; list as its argument and returns the second element of
;; that list'' over ``this function gets an argument, x,
;; and returns the car of the cdr of x.''  The key is to
;; describe the function's behavior holistically.
;; 
;; Now give this mystery function a try.

(define (mystery x)
  (cond
    ((null? x)        x)
    ((null? (car x))  (mystery (cdr x)))
    ((pair? (car x))  (append (mystery (car x)) (mystery (cdr x))))
    (else             (cons (car x) (mystery (cdr x))))))

;; 
;; This function takes a list as its argument and removes all of the
;; sub-lists and empty lists from the list, so that all of the elements
;; are still in the list but the structure is completely linear
;; 


;;;;;
;;;;; Exercise 4
;;;;;

;; Here is an incomplete implementation of restrict:

(define (restrict enzyme dna)
  (restrict-position-helper enzyme dna 0))

(define (restrict-position-helper enzyme dna position)
  (cond
    ((null? dna) '())
    ((restrict-mystery-helper? enzyme dna)
      (cons position
            (restrict-position-helper enzyme (cdr dna) (+ 1 position))))
    (else (restrict-position-helper enzyme (cdr dna) (+ 1 position)))))

;;; (a)
;
;; What must restrict-mystery-helper? do?  Describe it.

;; 
;; restrict-mystery-helper is a predicate that returns #t if the
;; enzyme can cut the dna at the current position.
;; 

;;; (b)
;
;; Write restrict-mystery-helper?.
;
(define (restrict-mystery-helper? enzyme dna)
  (cond
    ((and (null? dna) (not (null? enzyme))) #f)
    ((null? enzyme) #t)
    ((equal? (car enzyme) (car dna))
       (restrict-mystery-helper? (cdr enzyme) (cdr dna)))
    (else #f)))

;;; (c)
;
;; Write tests for restrict-mystery-helper?.  Your tests
;; should cover each branch in the code.  Uncomment the
;; do-tests when you're ready for it to run.
;
 (do-tests (restrict-mystery-helper? '(a b c) '(a b c)) #t
           (restrict-mystery-helper? '(a b c) '(a b))   #f
           (restrict-mystery-helper? '(a b) '(a b c))   #t
           (restrict-mystery-helper? '(c d) '(a b c))   #f
           (restrict-mystery-helper? '(b c) '(a b c))   #f)



;;;;;
;;;;; Exercise 5
;;;;;

;; Below is a function that takes the four functions in the
;; dictionary interface as arguments and runs several tests
;; using them.  ADD THREE TESTS TO THE do-tests BELOW.


(define (test-dictionary-functions new extend lookup keys)
  (do-tests
    (keys (new))                                        '()
    (keys (extend (new) 'a 'b))                         '(a)
    (lookup (extend (new) 'a 'b) 'a)                    'b
    (lookup (extend (extend (new) 'a 'b) 'c 'd) 'a)     'b
    (keys (extend (extend (new) 'a 'b) 'c 'd))          '(a c)
    (lookup (extend (new) 'a 'b) 'c)                    '()
    (lookup (extend (extend (new) 'a 'b) 'c 'd) 'c)     'd))



;; Here are implementations of dict-new and dict-extend for
;; dictionaries-as-lists:

(define (dictlist-new) '())

(define (dictlist-extend dl key value)
  (cond
    ((null? dl)          (list (cons key value)))
    ((equal? key (caar dl)) (cons (cons key value) (cdr dl)))
    (else
      (cons (car dl) (dictlist-extend (cdr dl) key value)))))


;;;;;
;;;;; Exercise 6
;;;;;


;;; (a)
;
;; Write an association list version of dict-keys; call it
;; dictlist-keys.
;
(define (dictlist-keys dl)
  (cond
    ((null? dl) '())
    (else (cons (caar dl) (dictlist-keys (cdr dl))))))

;;; (b)
;
;; Write dictlist-lookup.
;
(define (dictlist-lookup dl key)
  (cond
    ((null? dl) '())
    ((equal? key (caar dl)) (cdar dl))
    (else (dictlist-lookup (cdr dl) key))))


;; Now that we have a complete list-based
;; dictionary implementation, we can test it with
;; test-dictionary-functions.  Uncomment it when you're ready
;; for the tests to run:
;; 
 (test-dictionary-functions dictlist-new dictlist-extend
                            dictlist-lookup dictlist-keys)


;;;;;
;;;;; Exercise 7
;;;;;

;; Write tests for dictlist-keys.  Uncomment them when you're
;; ready for them to run.
;
 (do-tests
  (dictlist-keys (dictlist-new))                  '()
  (dictlist-keys (dictlist-extend (dictlist-extend
    (dictlist-new) 'a 'b) 'c 'd))      '(a c)
  (dictlist-keys (dictlist-extend (dictlist-extend
    (dictlist-new) 'c 'd) 'a 'b))      '(c a))


;; Our second representation for dictionaries is binary
;; trees.  You may recall that a binary tree is a linked data
;; structure; we will define a tree to be either the empty
;; tree, represented as (), or a node containing an object
;; and left and right subtrees, represented as a list:

(define (tree-make-empty) '())
(define (tree-empty? tree) (null? tree))
(define (tree-make-node obj left right) (list obj left right))
(define (tree-obj tree) (car tree))
(define (tree-left tree) (cadr tree))
(define (tree-right tree) (caddr tree))

;; Moving on to dictionaries, then, we need a way to store
;; both the key and the value in the object slot of a tree
;; node.  Since binary trees depend on having an ordering
;; for their keys, we'll also need a way to order symbols.

(define (dicttree-make-node key value left right)
  (tree-make-node (cons key value) left right))
(define (dicttree-node-key dt) (car (tree-obj dt)))
(define (dicttree-node-value dt) (cdr (tree-obj dt)))
(define (symbol<? a b)
  (string<? (symbol->string a) (symbol->string b)))

;; Here are dict-new and dict-lookup for dictionary trees:

(define (dicttree-new) (tree-make-empty))

(define (dicttree-lookup dt key)
  (cond
    ((tree-empty? dt)            '())
    ((equal? key (dicttree-node-key dt)) (dicttree-node-value dt))
    ((symbol<? key (dicttree-node-key dt))
      (dicttree-lookup (tree-left dt) key))
    (else
      (dicttree-lookup (tree-right dt) key))))


;;;;;
;;;;; Exercise 8
;;;;;


;;; (a)
;
;; Write dicttree-keys.  Your code for manipulating the
;; dictionary tree should be written in terms of the tree-
;; and dicttree-node- abstraction layer, not cars and cdrs.
;
(define (dicttree-keys dt)
  (mystery
    (cond
      ((tree-empty? dt) '())
      (else (cons (dicttree-keys (tree-left dt))
                  (cons (dicttree-node-key dt)
                    (dicttree-keys (tree-right dt))))))))

;;; (b)
;
;; Write dicttree-extend.  (Don't worry about tree balancing.)
;
(define (dicttree-extend dt key value)
  (cond
    ((tree-empty? dt) (dicttree-make-node key value '() '()))
    ((equal? key (dicttree-node-key dt))
     (dicttree-make-node
      key value (tree-left dt) (tree-right dt)))
    ((symbol<? key (dicttree-node-key dt))
     (dicttree-make-node
      (dicttree-node-key dt) (dicttree-node-value dt)
      (dicttree-extend (tree-left dt) key value)
      (tree-right dt)))
    (else (dicttree-make-node
      (dicttree-node-key dt) (dicttree-node-value dt)
      (tree-left dt)
      (dicttree-extend (tree-right dt) key value)))))


;; Now that we have a tree-based dictionary implementation,
;; we can test it.  Uncomment when you're ready for the tests
;; to run:
;; 
 (test-dictionary-functions dicttree-new dicttree-extend
                            dicttree-lookup dicttree-keys)


;;;;;
;;;;; Exercise 9
;;;;;

;; About how long did this assignment take you to complete?
;; How confident are you about your code? Answering the
;; question is worth 1 point; what your answer actually is
;; will not affect your grade provided that you answer it.
;; However, it provides us with feedback that will help us
;; improve this course for future years.  
;; 
;; The assignment took me a total of about 4 hours, but I
;; felt quite confident and comfortable with the coding.
