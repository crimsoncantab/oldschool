;; CS51 Assignment 0
;;
;; File:          asst0.scm
;; Author:        Loren McGinnis <mcginn@fas.harvard.edu>
;; Collaborators: None
;;
;; Collaboration is not permitted on this assignment.  By submitting
;; this assignment, you certify that you have followed the CS51
;; collaboration policy.  If you are unclear on the CS51 collaboration
;; policy, please ask your TF or the head TF.


;; NB: When you start up the scheme interpreter while loading this file,
;; you might see some "Oops!" messages.  This is because your functions
;; aren't defined yet.


;;;;;
;;;;; Exercise 1
;;;;;



;;;;;
;;;;; Exercise 2
;;;;;

; Determine the value of each these Scheme expressions and
; type that value in the provided spaces.  You may want to
; check your work in a Scheme interpreter.  Understanding how
; Scheme arrives at these results will help prepare you for
; the rest of the assignment.

;;; (a)
;
; (if (symbol? 'foo) '() (+ 1 2 3 4))
;
; => ()

;;; (b)
;
; (if (> 10 (max 5 6 7)) 'High)
;
; => high

;;; (c)
;
; (equal? 'foo #t)
;
; => #f

;;; (d)
;
; (equal? 'foo #f)
;
; => #f

;;; (e)
;
; (equal? 'foo "foo")
;
; => #f

;;; (f)
;
; (begin (display 75) 5)
;
; => 755


;;;;;
;;;;; Exercise 3
;;;;;

; Look up the function exact->inexact in R5RS, the Scheme Reference
; (actually, it's the entire specification for the language) linked
; from the website.  Describe what it does in English.  A good answer will
; explain the idea of exact and inexact numbers.
;
; exact->inexact takes an exact number, which is a number that represents
; a quantity exactly, such as a fraction, and gives the closest inexact
; number, like a decimal approximation, to it.
; 


;;;;;
;;;;; Exercise 4
;;;;;


;;; (a)
;
; Define a predicate (sum-squares? a b c) that returns #t if
; a^2 + b^2 = c^2 and #f otherwise. Your solution
; must handle non-number inputs.
;
(define (sum-squares? a b c)
  (if (number? a) (if (number? b) (if (number? c)
       (if (= (+ (* a a) (* b b)) (* c c)) #t #f)
         #f) #f) #f)
)



;;; (b)
;
; Define a predicate right-triangle? that takes the lengths of the sides
; of a triangle in any order and returns whether they specify a right
; triangle. Your code should never return true for a geometrically invalid
; triangle.
;
(define (right-triangle? a b c)
  (cond ((sum-squares? a b c) #t)
        ((sum-squares? b c a) #t)
        ((sum-squares? c a b) #t)
        (else #f)
        )
)


;;;;;
;;;;; Exercise 5
;;;;;

; Define a function print-cubes that takes one argument n >= 1 and
; print a list of cubes of the integers from n down to 1, in
; descending order, each on an individual line.  Your function
; should return the empty list in all cases. (Hint: look up display in the
; Scheme Reference.  How can you both display a value and return a value?)
; Your function should handle floating point numbers by truncating the
; input (hint: look up floor).
;
(define (print-cubes n)
  (if (number? n)
    (if (< (floor n) 1) '()
        (begin 
          (display (inexact->exact (* (floor n) (floor n) (floor n))))
          (newline)
          (print-cubes (- n 1))
        )
     ) 
   '()
  )
)

; 
; Here are some test expressions.  When you run "make tests",
; this file will be run by a scheme interpreter and the results
; of these tests will be both saved in the files asst0.sout
; and printed to the screen.  In future assignments, we will
; provide you a framework for writing more complex tests.  For
; now, just check if the output is what you expect.

(print-cubes 5)
(print-cubes 1)

(sum-squares? 3 4 5)
(sum-squares? 1 2 4)
(sum-squares? 5 4 3)

(right-triangle? 3 4 5)
(right-triangle? 1 2 4)
(right-triangle? 5 4 3)


;; Please write your answers to the C Review questions in answers.txt!


;;;;; End of Assignment 0
