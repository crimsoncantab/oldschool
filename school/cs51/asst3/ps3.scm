

;; CS51 Assignment 3
;; 
;; File:          asst3.scm
;; Author:        Alex Yang <alexyang@fas.harvard.edu>
;; Partner:       Loren McGinnis <mcginn@fas.harvard.edu>
;; Collaborators: None
;; 
;; Collaboration is required on this assignment.
;; By submitting this assignment, you certify that you have
;; followed the CS51 collaboration policy.  If you are unclear
;; on the CS51 collaboration policy, please ask your TF or
;; the head TF.

;;;;;
;;;;; Exercise 1
;;;;;

; Email your design document to your TF by Monday, 25
; February 2008 at 5:00 PM.

;;;;;
;;;;; Exercise 2
;;;;;


;;; (a)
;
; For the interface to implement undirected graphs, what
; contract needs to exists between graph-connect and
; graph-adjacent??
;
; When graph-connect is called, not only does n1 need to be connected to n2,
; but n2 needs to be connected to n1. Consequently, (graph-adjacent g n1 n2)
; and (graph-adjacent g n2 n1) both should return #t. Therefore, graph-connect 
; and graph-adjacent both need to use the same data structures.
;

;;; (b)
;
; Oops!  The interface above is missing a method that
; is necessary if we want to be able to represent every
; undirected graph.  Can you figure out what kind of graph
; cannot be created with the interface above?  What is
; missing?  Include it in your implementation.
;
; The given methods in the interface above are unable to create a graph with
; nodes that are isolated by themselves. To fix this, we can create a method
; called (graph-add g n1) which returns a graph like g but with node n1 added.
;

;;; (c)
;
; Implement the interface for graphs.

;implementation of dictionary as list from Asst1
(define (dictlist-new) '())

(define (dictlist-extend dl key value)
  (cond
    ((null? dl)          (list (cons key value)))
    ((equal? key (caar dl)) (cons (cons key value) (cdr dl)))
    (else
      (cons (car dl) (dictlist-extend (cdr dl) key value)))))

(define (dictlist-keys dl)
  (cond
    ((null? dl) '())
    (else (cons (caar dl) (dictlist-keys (cdr dl))))))

(define (dictlist-lookup dl key)
  (cond
    ((null? dl) '())
    ((equal? key (caar dl)) (cdar dl))
    (else (dictlist-lookup (cdr dl) key))))

;standard reduce function
(define (reduce func base lst)
   (if (null? lst)
       base
       (func (car lst) (reduce func base (cdr lst)))))

;returns #t if element is in lst, else #f
(define (contains? lst element)
  (reduce (lambda (x y) (or (equal? element x) y)) #f lst))

(define (graph-empty)
  (dictlist-new))

;makes a one-way association from n1 to n2
(define (graph-extend g n1 n2)
  (if (contains? (graph-neighbors g n1) n2) g
      (dictlist-extend g n1 (cons n2 (graph-neighbors g n1)))))

(define (graph-connect g n1 n2)
  (graph-extend (graph-extend g n1 n2) n2 n1))

(define (graph-adjacent? g n1 n2)
  (contains? (dictlist-lookup g n1) n2))
  
(define (graph-neighbors g n)
  (dictlist-lookup g n))

(define (graph-nodes g)
  (dictlist-keys g))

(define (graph-add g n)
  (if (contains? (graph-nodes g) n) g
      (dictlist-extend g n '())))

; You don't really need templates for this, do you?

(do-tests
  (contains? '(1 2 3) 2)
                        #t
  (contains? '(1 2 3) 4)
                        #f
  (contains? '() 1)
                        #f
  (contains? '(1 2 3) '())
                        #f
  (contains? '() '())
                        #f
  (graph-empty)
                        (dictlist-new)
  (graph-neighbors (graph-empty) 3)
                        '()
  (graph-neighbors (graph-extend (graph-empty) 1 2) 1)
                       '(2)
  (graph-neighbors (graph-connect (graph-connect (graph-empty) 1 2) 1 3) 1)
                       '(3 2)
  (graph-neighbors (graph-extend (graph-empty) 1 2) 2)
                       '()
  (graph-neighbors (graph-connect (graph-connect (graph-empty) 1 2) 1 3) 2)
                       '(1)
  (graph-neighbors (graph-connect (graph-connect (graph-empty) 1 2) 1 3) 4)
                       '()
  (graph-nodes  (graph-connect (graph-connect (graph-empty) 1 2) 1 3))
                       '(1 2 3)
  (graph-nodes (graph-empty))
                       '()
  (graph-nodes (graph-connect (graph-connect (graph-connect (graph-empty) 1 2) 1 3) 1 3))
                       '(1 2 3)
  (graph-adjacent? (graph-connect (graph-connect (graph-empty) 1 2) 1 3) 1 3)
                       #t
  (graph-adjacent? (graph-connect (graph-connect (graph-empty) 1 2) 1 3) 2 3)
                       #f
  (graph-adjacent? (graph-empty) 1 2)
                       #f
  (graph-adjacent? (graph-connect (graph-connect (graph-empty) 1 2) 1 3) 1 4)
                       #f
  (graph-neighbors (graph-add (graph-connect (graph-connect (graph-empty)
                      1 2) 1 3) 4) 4)
                       '()
  (graph-neighbors (graph-add (graph-connect (graph-empty) 1 2) 2) 2)
                       '(1))
;;;;;
;;;;; Exercise 3
;;;;;

; Write a function (spork-schedule talk-pairs timeslots) that
; returns an association list mapping talks to timeslots if a
; valid schedule exists, or #f otherwise.  Think carefully
; about modularity---you may find code that implements
; non--domain-specific functionality to be useful later.
; Your function must use recursive backtracking, and it
; should treat graphs as an opaque, abstract data type.

;returns nth element of lst (indexing starts at 0)
(define (listref lst n)
  (cond
   ((null? lst) '())
   ((= n 0) (car lst))
   (else (listref (cdr lst) (- n 1)))))

;standard filter function
(define (filter pred? lst)
  (cond
    ((null? lst) '())
    ((pred? (car lst)) (cons (car lst) (filter pred? (cdr lst))))
    (else (filter pred? (cdr lst)))))

; removes all occurrences of element from lst
(define (remove-element element lst)
  (filter (lambda (x) (not (equal? element x))) lst))

; removes all occurrences of elements of lst1 from lst2
(define (remove-list lst1 lst2)
  (reduce (lambda (x y) (remove-element x y)) lst2 lst1))

; creates a graph with the associations given in talk-pairs
(define (create-graph talk-pairs)
  (reduce (lambda (x y) (graph-connect y (car x) (cdr x)))
          (graph-empty) talk-pairs))

; creates an empty schedule, where each node is associated with '()
(define (schedule-new graph)
  (reduce (lambda (x y) (cons (cons x '()) y)) '() (graph-nodes graph)))

; looks up the color associated with node
(define (schedule-lookup schedule node)
  (cond
   ((null? schedule)                   '())
   ((equal? node (caar schedule))      (cdar schedule))
   (else
      (schedule-lookup (cdr schedule) node))))

; associates a given node with a given color
(define (schedule-extend schedule node color)
  (cond
    ((null? schedule)         (list (cons node color)))
    ((equal? node (caar schedule)) (cons (cons node color) (cdr schedule)))
    (else
      (cons (car schedule) (schedule-extend (cdr schedule) node color)))))

; returns a list of available colors for a given node
(define (available-colors graph node colors schedule)
  (remove-list (reduce (lambda (x y) (cons (schedule-lookup schedule x) y))
                            '() (graph-neighbors graph node)) colors))

; returns nth node from graph created with talk-pairs, or () if not possible
(define (pick-node graph n)
  (if (<= (length (graph-nodes graph)) n)
      '()
      (list-ref (graph-nodes graph) n)))
      
(define (spork-schedule talk-pairs timeslots)
  (color-graph (create-graph talk-pairs) timeslots))

(define (update-schedule graph noden color schedule)
  (schedule-extend schedule (listref (graph-nodes graph) noden)  color))

(define (color-graph graph colors)
  (color-node graph colors 0 (schedule-new graph)))

;returns the element in lst that follows given element
;returns () if it doesn't exist
(define (next-element lst element)
  (cond
    ((null? lst) '())
    ((null? element)    (car lst))
    ((equal? element (car (reverse lst)))     '())
    ((equal? element (car lst)) (cadr lst))
    (else (next-element (cdr lst) element))))

;returns the next available color for a given node      
(define (next-color graph colors node schedule)
  (next-element (available-colors graph node colors schedule)
                (schedule-lookup schedule node)))

;
(define (color-node graph colors noden schedule)
  (cond
   ((>= noden (length (graph-nodes graph))) schedule)
   ((null? (next-color graph colors (pick-node graph noden) schedule))
    (if (= noden 0) #f
	       (color-node graph colors (- noden 1)
		    (update-schedule graph noden '() schedule))))
   (else (color-node graph colors (+ noden 1)
		     (update-schedule graph noden 
				      (next-color graph colors
				       (pick-node graph noden) schedule)
				      schedule)))))


;;;;;
;;;;; Exercise 4
;;;;;

; We have provided a number of tests for spork-schedule,
; but you should add more tests to prove to us that your
; program works.  You must include a test case for which your
; particular implementation of graph coloring would have to
; backtrack at some point.  You should also write thorough
; tests for any helper functions you write, including the
; graph interface.  Thorough tests would test each path in
; your helper functions.

; (do-tests expr => function) means to pass the result of
; expr to function, and consider the test to have failed
; if the function returns #f.  Some of these tests check
; whether a schedule was found, but not what it was.
; The graphs below are from the figures in the printed
; version of the assigment.


 (do-tests
   (spork-schedule '() '(red green))
                               '()
   (spork-schedule '((a . b) (b . c) (b . d)) '())
                               #f
   (spork-schedule '((a . b) (b . c) (b . d)) '(red))
                               #f
   (spork-schedule '((a . b) (b . c) (b . d)) '(red green))
                               (list (cons 'b 'red) (cons 'd 'green) (cons 'c 'green)
                               (cons 'a 'green))
   (spork-schedule '((a . b) (b . c) (b . d) (c . d)) '(red green))
                               #f
   (spork-schedule '((a . b) (b . c) (b . d) (c . d)) '(red green blue))
                         (list (cons 'c 'red) (cons 'd 'green)
			       (cons 'b 'blue) (cons 'a 'red))
   (spork-schedule '((a . b) (a . c) (b . c) (b . d) (c . d)
                     (d . e) (d . f) (e . f)) '(red))
                               #f
   (spork-schedule '((a . b) (a . c) (b . c) (b . d) (c . d)
                     (d . e) (d . f) (e . f)) '(red green))
                               #f
   (spork-schedule '((a . b) (a . c) (b . c) (b . d) (c . d)
                     (d . e) (d . f) (e . f)) '(red green blue))
		     (list (cons 'e 'red) (cons 'f 'green) (cons 'd 'blue)
			   (cons 'c 'red) (cons 'b 'green) (cons 'a 'blue))
   (spork-schedule '((a . b) (a . c) (b . c) (b . d) (c . d)
                     (d . e) (d . f) (e . f) (g . h)) '(red green blue))
                     (list (cons 'g 'red) (cons 'h 'green)
			   (cons 'e 'red) (cons 'f 'green) (cons 'd 'blue)
			   (cons 'c 'red) (cons 'b 'green) (cons 'a 'blue))
   (spork-schedule '((a . b) (b . c) (b . d) (e . f) (e . g) (f . g))
		   '(red green))
                               #f
;back-tracking required:
   (spork-schedule '((c . b) (c . d) (b . a) (e . d)) '(red green))
                     (list (cons 'e 'red) (cons 'd 'green) (cons 'b 'green)
			   (cons 'a 'red) (cons 'c 'red))

   (next-element '(a b c d) 'a)            'b
   (next-element '(a b c d) 'b)            'c
   (next-element '(a b c d) 'd)            '()
   (next-element '() 'a)                   '()

   (schedule-extend '((a . '())) 'a  'red) 
                (list (cons 'a 'red))
   (schedule-extend '((a . ()) (b . ()) (c . ())) 'b 'red)
                (list (cons 'a '()) (cons 'b 'red) (cons 'c '()))
   (schedule-extend '() 'b 'red)
                (list (cons 'b 'red))

   (listref '(1 2 3 4 5) 0)                 1
   (listref '(1 2 3 4 5) 5)                 '()
   (listref '(1 2 3 4 5) 2)                 3
   
   (remove-list '() '())                    '()
   (remove-list '() '(1 2 3))               '(1 2 3)
   (remove-list '(1 2 3) '(1 2 3))          '()
   (remove-list '(1) '(2 3))                '(2 3)
   (remove-list '(1 2) '(1 1 2 2 3 1)) '(3))


;;;;;
;;;;; Exercise 5
;;;;;


;;; (a)
;
; One of your classmates came up with a novel coloring
; algorithm that, for commonly occurring scheduling
; conflicts, runs in polynomial time.They always say that at
; Harvard you learn the most from your classmates!  Suppose
; that you've been given a function (color-in-polynomial
; graph colorlist).  How much of your code would you have
; to change for spork-schedule to use the new coloring
; routine in place of your own?  Type your brief answer
; as a comment; writing the new spork-schedule in terms of
; color-in-polynomial (in a comment) is likely to receive
; full credit.
;
; We would not have to change much code
; (define (spork-schedule talk-pairs timeslots)
;   (color-in-polynomial (create-graph talk-pairs) timeslots))

;;; (b)
;
; Spork is having trouble making a seating chart for their
; dinner on the first night.  A number of spork sellers
; are not on speaking terms, and putting a feuding pair at
; the same table could be explosive.  If Spork prepares a
; list of who is mad at whom and decides how many tables
; they would like, how long would it take for you to write a
; function to perform table assignments?  Type your answer in
; a comment; then write (spork-table-assignment feuding-pairs
; table-list).

;It should not take very much coding at all, merely a call
;to the spork-schedule function.
;
;(define (spork-table-assignment feuding-pairs table-list)
;  (spork-schedule feuding-pairs table-list))



;;;;;
;;;;; WHEN YOU ARE FINISHED
;;;;;

; About how long did this take your group to complete?
; How confident are you with your code?  How long did you
; spend on the assignment, and how long did your partner
; spend on this assignment?  Your answers will not affect
; your grade.  Type your answer in a comment below.
;
;
