;; CS51 Assignment 4
;; 
;; File:          asst4.scm
;; Author:        Loren McGinnis <mcginn@fas.harvard.edu>
;; Collaborators: None
;; 
;; Collaboration is not permitted on this assignment.
;; By submitting this assignment, you certify that you have
;; followed the CS51 collaboration policy.  If you are unclear
;; on the CS51 collaboration policy, please ask your TF or
;; the head TF.

;standard reduce function
(define (reduce func base lst)
   (if (null? lst)
       base
       (func (car lst) (reduce func base (cdr lst)))))

;standard filter function
(define (filter pred? lst)
  (cond
    ((null? lst) '())
    ((pred? (car lst)) (cons (car lst) (filter pred? (cdr lst))))
    (else (filter pred? (cdr lst)))))



;;;;;
;;;;; Exercise 1
;;;;;

;; Write a function
;; dilemma-table that takes two inputs, player 1's move and
;; player 2's move, and returns a dotted pair of the scores for the
;; prisoner's dilemma game table given in the problem introduction.  For
;; instance, (dilemma-table 's 't)) should evaluate to
;; (10 . 0).
;; 

(define (dilemma-table p1 p2)
  (cond
     ((equal? p1 p2) (if (equal? p1 's) (cons 1 1) (cons 5 5)))
     ((equal? p1 't) (cons 0 10))
     (else (cons 10 0))))



;;;;;
;;;;; Exercise 2
;;;;;


;;; (a)
;
;; Write the all-testify strategy.  For instance,
;; (all-testify dilemma-table '(s t s) '(t s s)) should
;; evaluate to 't.

(define (all-testify dilemma-table p1 p2)
  't)


;;; (b)
;
;; Write the tit-for-tat strategy.
(define (tit-for-tat dilemma-table p1 p2)
  (if (null? p1) 's (car p1)))


;;; (c)
;
;; Write the go-by-majority strategy.

;returns the number of occurences of element in lst
(define (list-freq lst element)
  (reduce (lambda (x y) (if (equal? element x) (+ 1 y) (+ 0 y))) 0 lst))


(define (go-by-majority game-table p1 p2)
  (if (> (list-freq p1 't) (list-freq p1 's)) 't 's))



;;;;;
;;;;; Exercise 3
;;;;;

;; Write a function play-repeated-game that takes four inputs: a
;; game table, player 1's strategy, player 2's strategy, and an integer
;; greater than zero number of times to repeat, and returns a dotted pair
;; containing the the first player's total score as the car and
;; the second player's total score as the cdr.

;returns a pair where the first player's history of moves is the car
;and the second player's history is the cdr, given strategies for both,
;a game table, and the number of times to play the game
(define (generate-moves game-table p1s p2s p1 p2 n)
  (if 
   (<= n 0) (cons p1 p2)
   (generate-moves game-table p1s p2s 
                   (cons (p1s game-table p2 p1) p1)
                   (cons (p2s game-table p1 p2) p2)
                   (- n 1))))

;combines two lists into one list of pairs, until one of the lists is empty
(define (pair-up lst1 lst2)
  (if
   (or (null? lst1) (null? lst2) (not (pair? lst1)) (not (pair? lst2))) '()
   (cons (cons (car lst1) (car lst2)) (pair-up (cdr lst1) (cdr lst2)))))

;pairs up car of list with cdr
(define (pair-up-helper lst)
  (if
   (null? lst) '()
  (pair-up (car lst) (cdr lst))))

(define (play-repeated-game game-table p1s p2s n)
  (reduce (lambda (x y) (cons (+ (car (game-table (car x) (cdr x))) (car y))
                              (+ (cdr (game-table (car x) (cdr x))) (cdr y)))) (cons 0 0)
          (pair-up-helper (generate-moves game-table p1s p2s '() '() n))))

;;;;;
;;;;; Exercise 4
;;;;;


;;; (a)
;
;; Write a function combine that takes a
;; list of strategies and a list of floating-point numbers
;; representing a probability distribution
;; and returns a new strategy that randomly plays
;; each input strategy proportionally to its
;; frequency in the distribution.  For instance, (combine
;;   (list tit-for-tat go-by-majority) '(.7 .3)) should return a strategy
;; that, on average, plays tit-for-tat 70 percent of the time,
;; and plays go-by-majority 30 percent of the time. 

;given a number, picks a strategy from a list of pairs depending upon the number
(define (pick-strat strat-pairs n)
  (cond
    ((null? strat-pairs) '())
    ((> (cdar strat-pairs) n) (caar strat-pairs))
    (else (pick-strat (cdr strat-pairs) (- n (cdar strat-pairs))))))

(define (combine slist plist)
  (lambda (x y z) ((pick-strat (pair-up slist (map (lambda (a) (* 100 a)) plist)) 
                               (random 100)) x y z)))

;;; (b)
;
;; In a comment, explain with a few sentences why your function has the
;; correct average behavior.
;;
;; The function converts the probabilities into 2 digit numbers.  The random
;; number that it takes is also between 0 and 99.  In deciding upon a strategy,
;; the function subtracts each proceeding probability from the random number
;; until the next probability is greater than what is remaining.  Larger numbers
;; will make the function pick strategies that are later in the list, and vice-versa.
;; There is no bias because large numbers have the same probability of being chosen
;; as small numbers.

;;; (c)
;
;; Use play-repeated-game to test combine by writing
;; three tests that convincingly illustrate the correctness of combine over
;; time.

(define (all-silent dilemma-table p1 p2)
  's)

(play-repeated-game dilemma-table 
                    (combine (list all-testify tit-for-tat) '(.5 .5)) all-silent 100)
(play-repeated-game dilemma-table 
                    (combine (list all-testify tit-for-tat) '(0 1)) tit-for-tat 10)
(play-repeated-game dilemma-table 
                    (combine (list all-testify all-silent) '(.5 .5)) tit-for-tat 100)

;;;;;
;;;;; Exercise 5
;;;;;

;; Write a single strategy that will compete in both of tournaments.
;; Prizes will be rewarded.  Name your function tournament-name,
;; where name is your FAS email username.

(define (tournament-mcginn dilemma-table opp me)
  (cond
    ((<= (car (dilemma-table 't 't)) (car (dilemma-table 's 's)))
     (cond
       ((<= (car (dilemma-table 't 's)) (cdr (dilemma-table 't 's))) 't)
       ((null? opp) 't)
       (else (car opp))))
    ((>= (car (dilemma-table 't 's)) (cdr (dilemma-table 't 's))) 's)
    ((null? opp) 's)
    (else (car opp))))

;; Here are two stream examples:
(define ones (cons 1 (delay ones)))
(define (count-from n)
  (cons n (delay (count-from (+ 1 n)))))


;; stream-ref: Stream of X x Integer -> X
;; To get the nth element of a stream.
(define (stream-ref stream n)
  (if (zero? n)
    (car stream)
    (stream-ref (force (cdr stream)) (- n 1))))


;;;;;
;;;;; Exercise 6
;;;;;

;; Write a function (stream->list stream n) that returns the
;; first n elements of stream as a list.  stream->list must
;; run in O(n).

(define (stream->list stream n)
  (if (zero? n)
      '()
      (cons (car stream) (stream->list (force (cdr stream)) (- n 1)))))

;;;;;
;;;;; Exercise 7
;;;;;

;; stream-map: (X -> Y) x Stream of X -> Stream of Y
;; To map a function over an infinite stream.
(define (stream-map f stream)
  (cons
    (f (car stream))
    (delay (stream-map f (force (cdr stream))))))

;;; (a)
;
;; Write a function (stream-max s1 s2) that takes two
;; streams and returns a new stream of the maximum of the
;; first elements of s1 and s2, followed by the maximum of
;; the second element of each, and so on.  (For example,
;; s1 and s2 might represent simultaneous rolls of two dice,
;; and we want a stream representing the maximum in each pair
;; of rolls.)  You may assume that both streams are infinite.

;returns max of x and y
(define (max? x y)
  (if (> x y) x y))

(define (stream-max s1 s2)
  (cons (max? (car s1) (car s2)) 
        (delay (stream-max (force (cdr s1))
                           (force (cdr s2))))))
                                                    

;;; (b)
;
;; Write a function (stream-unfold start next) that returns an
;; infinite stream start, (next start), (next (next start)),
;; (next (next (next start))), . . . .

(define (stream-unfold start next)
  (cons start (delay (stream-unfold (next start) next))))

;;;;;
;;;;; Exercise 8
;;;;;

;; The function (within stream eps) follows stream until it
;; finds successive values that differ by less then eps, at
;; which point it returns the second of the successive values.
;; (Note that if the stream fails to converge, within will
;; not terminate.)  Write within.

(define (sqrt-stream x guess)
(let ((next (/ (+ guess (/ x guess)) 2)))
(cons next (delay (sqrt-stream x next)))))

(define (within stream eps)
  (if
   (< (abs (- (car stream) (car (force (cdr stream))))) eps)
   (car (force (cdr stream)))
   (within (force (cdr stream)) eps)))

;;;;;
;;;;; Exercise 9
;;;;;

;; The golden ratio may be defined in numerous ways.  It is a
;; solution to the equation f = (f + 1)/f, and we can compute
;; it using a series of approximations f[n] = 1 + 1/f[n-1]
;; with f[0] = 1.  Define golden-ratio to be the stream f_0,
;; f_1, f_2, . . . .

(define (golden-ratio)
  (stream-unfold 1.0 (lambda (x) (+ 1 (/ 1 x)))))


;;;;;
;;;;; Exercise 10
;;;;;

;; The function e^x may be computed by the power series
;; Sigma x^i/i! from i=0.  Write a function (e-stream x)
;; that returns a stream of approximations for e^x like 1,
;; 1 + x, 1 + x + x^2/2, . . ., in which each successive
;; approximation contains one more term of the power series.
;; Note that, since e-stream is applied to a particular x,
;; the returned stream is a sequence of improving numbers.

;computes factorial
(define (fact n)
  (if (< n 2) 1 
      (* n (fact (- n 1)))))

;calculate's the ith term of the summation for e
(define (calc-e-term x i)
  (if (< i 1) 1.0
      (+ (calc-e-term x (- i 1)) (/ (expt x i) (fact i)))))

(define (e-stream-helper x i)
  (cons (calc-e-term x i) (delay (e-stream-helper x (+ i 1)))))

(define (e-stream x)
  (e-stream-helper x 0))


;helper functions for testing
(define from1 (stream-unfold 1 (lambda (x) (+ 1 x))))
(define squares (stream-map (lambda (x) (* x x)) from1))
(define grault (stream-unfold 2 (lambda (x) (* 2 x))))


;testing
;(do-tests
;(dilemma-table 't 's)
;(cons 0 10)
;(dilemma-table 't 't)
;(cons 1 1)
;(dilemma-table 's 't)
;(cons 10 0)
;(dilemma-table 's 's)
;(cons 5 5)
;(all-testify dilemma-table '(s t s) '(t s s))
; 't
;(all-testify dilemma-table '() '())
; 't
;(tit-for-tat dilemma-table '(s t s) '(t s s))
;'s
;(tit-for-tat dilemma-table '(t t s) '(t s s))
;'t
;(tit-for-tat dilemma-table '() '())
;'s
;(list-freq '() 'a)
;0
;(list-freq '(a b a) 'a)
;2
;(list-freq '(b b b) 'a)
;0
;(list-freq '(b b b) '())
;0
;(go-by-majority  dilemma-table '(s t s) '(t s s))
;'s
;(go-by-majority  dilemma-table '(s t t) '(t s s))
;'t
;(go-by-majority  dilemma-table '() '())
;'s
;(generate-moves dilemma-table go-by-majority all-testify '() '() 10)
;'((t t t t t t t t t s) t t t t t t t t t t)
;(generate-moves dilemma-table go-by-majority tit-for-tat '() '() 10)
;'((s s s s s s s s s s) s s s s s s s s s s)
;(generate-moves dilemma-table all-testify tit-for-tat '() '() 10)
;'((t t t t t t t t t t) t t t t t t t t t s)
;(generate-moves dilemma-table all-testify tit-for-tat '() '() 0)
;'(())
;(generate-moves dilemma-table all-testify all-testify '(s s t) '(t s s) 0)
;'((s s t) t s s)
;(generate-moves dilemma-table all-testify all-testify '(s s t) '(t s s) 3)
;'((t t t s s t) t t t t s s)
;(generate-moves dilemma-table go-by-majority tit-for-tat '(s s t) '(t s s) 3)
;'((s s s s s t) s s s t s s)
;(generate-moves dilemma-table go-by-majority tit-for-tat '(s s t) '(t s t) 3)
;'((t s t s s t) s t s t s t)
;(pair-up '() '())
;'()
;(pair-up '(1) '(1))
;'((1 . 1))
;(pair-up '(1) '(1 2))
;'((1 . 1))
;(pair-up '(1 2 3) '(1 2))
;'((1 . 1) (2 . 2))
;(pair-up-helper '())
;'()
;(pair-up-helper '(a b))
;'()
;(pair-up-helper '(a (b ())))
;'()
;(pair-up-helper '((b ()) a))
;'((b . a))
;(pair-up-helper '((a b c d) a b c d))
;'((a . a) (b . b) (c . c) (d . d))
;(play-repeated-game dilemma-table all-testify all-testify 1)
;'(5 . 5)
;(play-repeated-game dilemma-table all-testify all-testify 3)
;'(15 . 15)
;(play-repeated-game dilemma-table all-testify tit-for-tat 3)
;'(10 . 20)
;(play-repeated-game dilemma-table all-testify all-testify 0)
;'(0 . 0)
;(play-repeated-game dilemma-table tit-for-tat go-by-majority 10)
;'(10 . 10)
;(pick-strat (list '(a . 10) '(b . 20) '(c . 30) '(d . 40)) 24)
;'b
;(pick-strat '() 24)
;'()
;(pick-strat '() 0)
;'()
;(pick-strat '((a . 100) (b . 0)) 0)
;'a
;(pick-strat '((a . 100) (b . 0)) 99)
;'a
;(pick-strat '((a . 50) (b . 50)) 0)
;'a
;(pick-strat '((a . 99) (b . 1)) 99)
;'b
;(pick-strat '((a . 50) (b . 50)) 50)
;'b
;(stream->list (count-from 0) 0)
;'()
;(stream->list (count-from 0) 5)
;'(0 1 2 3 4)
;(stream->list (stream-max ones (count-from 2)) 5)
;'(2 3 4 5 6)
;(stream->list (stream-max ones (count-from -1)) 5)
;'(1 1 1 2 3)
;(stream->list (stream-max ones (count-from 1)) 5)
;'(1 2 3 4 5)
;(stream->list from1 5)
;'(1 2 3 4 5)
;(stream->list squares 5)
;'(1 4 9 16 25)
;(stream-ref squares 19)
;400
;(stream->list (stream-max squares grault) 5)
;'(2 4 9 16 32)
;(stream->list (stream-unfold 1 (lambda (x) (* -1 x))) 5)
;'(1 -1 1 -1 1)
;(stream->list (stream-unfold 1 (lambda (x) (* -1 x))) 0)
;'()
;(stream->list (stream-max (stream-unfold 1 (lambda (x) (* -2 x))) ones)  10)
;'(1 1 4 1 16 1 64 1 256 1)
;(within (sqrt-stream 64.0 1) 10)
;10.474036101145005
;(within (sqrt-stream 64.0 1) 1)
;8.005147977880979
;(within (sqrt-stream 64.0 1) 0.1)
;8.000001655289593
;(within (sqrt-stream 64.0 1) 0.001)
;8.00000000000017
;(stream->list (golden-ratio) 6)
;'(1.0 2.0 1.5 1.6666666666666665 1.6 1.625)
;(stream->list (golden-ratio) 0)
;'()
;(stream-ref (golden-ratio) 40)
;1.618033988749895
;(within (golden-ratio) .001)
;1.6181818181818182
;(calc-e-term 1 10)
;2.7182818011463845
;(calc-e-term 1 100)
;2.7182818284590455
;(calc-e-term 1 1)
;2.0
;(calc-e-term 2 100)
;7.389056098930649
;(within (e-stream 1) .1)
;2.708333333333333
;(within (e-stream 1) .001)
;2.7182539682539684
;(within (e-stream 2) .001)
;7.388994708994708
;(stream-ref (e-stream 1) 20)
;2.7182818284590455)

;;;;;
;;;;; WHEN YOU ARE FINISHED
;;;;;

;; About how long did this take you to complete?  How
;; confident are you in your code?  Your answer will not
;; affect your grade.  Type your answer in a comment below.
;
; This assignment took me about 6 hours of total work to complete
; I have done quite a lot of extensive testing with my code,
; so I feel very confident with it.

