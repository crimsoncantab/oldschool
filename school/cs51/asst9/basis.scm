;; Standard basis for scheme51

(define (quit)
  (display 'bye)
  (space)
  (display 'bye)
  (newline)
  (exit))

(define (not x)
  (if x #f #t))

(define (list . lst) lst)

(define (list? lst)
  (if (null? lst) #t
    (if (pair? lst) (list? (cdr lst))
      #f)))

(define (length lst)
  (if (null? lst) 0
   (+ 1 (length (cdr lst)))))

(define (!= n1 n2 . ns)
  (not (= n1 n2)))

(define (> . nums)
  (apply < (reverse nums)))

(define (>= . nums)
  (apply <= (reverse nums)))

(define (even? num)
  (= 0 (modulo num 2)))
(define (odd? num)
  (not (= 0 (modulo num 2))))
(define (zero? num)
  (= num 0))
(define (positive? num)
  (> num 0))
(define (negative? num)
  (< num 0))

(define (&& . lst)
  (if (null? lst) #t
    (if (null? (cdr lst)) (car lst)
      (if (car lst) (apply && (cdr lst))
        #f))))

(define (|| . lst)
  (if (null? lst) #f
    (if (car lst) (car lst)
      (apply || (cdr lst)))))

(define (eqv? a b)
  (if (&& (number? a) (number? b))
    (= a b)
    (eq? a b)))

(define (equal? a b)
  (if (&& (pair? a) (pair? b))
    (if (equal? (car a) (car b))
      (equal? (cdr a) (cdr b)) 
      #f)
    (eqv? a b)))

(define (foldl f zero lst)
  (if (null? lst) zero
    (foldl f (f (car lst) zero) (cdr lst))))

(define (foldr f zero lst)
  (if (null? lst) zero
    (f (car lst) (foldr f zero (cdr lst)))))

(define (foldr1 f lst)
  (if (null? (cdr lst)) (car lst)
    (f (car lst) (foldr1 f (cdr lst)))))

(define (foldl1 f lst)
  (foldl f (car lst) (cdr lst)))

(define (map1 f lst)
  (foldr (lambda (a b) (cons (f a) b)) '() lst))

(define (map f lst . lsts)
  (if (null? lst) '()
      (cons (apply f (car lst) (map1 car lsts))
            (apply map f (cdr lst) (map1 cdr lsts)))))

(define (append2 a b)
  (foldr cons b a))

(define (append . lsts)
  (foldr1 append2 lsts))

(define (reverse a)
  (foldl cons '() a))

(define (caar x) (car (car x)))
(define (cadr x) (car (cdr x)))
(define (cdar x) (cdr (car x)))
(define (cddr x) (cdr (cdr x)))
(define (caaar x) (car (car (car x))))
(define (caadr x) (car (car (cdr x))))
(define (cadar x) (car (cdr (car x))))
(define (caddr x) (car (cdr (cdr x))))
(define (cdaar x) (cdr (car (car x))))
(define (cdadr x) (cdr (car (cdr x))))
(define (cddar x) (cdr (cdr (car x))))
(define (cdddr x) (cdr (cdr (cdr x))))
(define (caaaar x) (car (car (car (car x)))))
(define (caaadr x) (car (car (car (cdr x)))))
(define (caadar x) (car (car (cdr (car x)))))
(define (caaddr x) (car (car (cdr (cdr x)))))
(define (cadaar x) (car (cdr (car (car x)))))
(define (cadadr x) (car (cdr (car (cdr x)))))
(define (caddar x) (car (cdr (cdr (car x)))))
(define (cadddr x) (car (cdr (cdr (cdr x)))))
(define (cdaaar x) (cdr (car (car (car x)))))
(define (cdaadr x) (cdr (car (car (cdr x)))))
(define (cdadar x) (cdr (car (cdr (car x)))))
(define (cdaddr x) (cdr (car (cdr (cdr x)))))
(define (cddaar x) (cdr (cdr (car (car x)))))
(define (cddadr x) (cdr (cdr (car (cdr x)))))
(define (cdddar x) (cdr (cdr (cdr (car x)))))
(define (cddddr x) (cdr (cdr (cdr (cdr x)))))

(define (list-ref n lst)
  (car (list-tail n lst)))

(define (list-tail n lst)
  (if (zero? n) lst
   (list-tail (- n 1) (cdr lst))))

(define (abs n)
  (if (negative? n) (- n) n))

(define (find pred lst)
  (if (null? lst) #f
    (if (pred (car lst)) lst
      (find pred (cdr lst)))))

(define (mem cmp item lst)
  (find (lambda (x) (cmp x item)) lst))

(define (memq item lst) (mem eq? item lst))
(define (memv item lst) (mem eqv? item lst))
(define (member item lst) (mem equal? item lst))

(define (ass cmp item lst)
  (car (find (lambda (x) (cmp (car x) item)) lst)))

(define (assq item lst) (ass eq? item lst))
(define (assv item lst) (ass eqv? item lst))
(define (assoc item lst) (ass equal? item lst))

(define (expt x y)
  (if (zero? y) 1
    (if (odd? y) (* x (expt x (- y 1)))
      ((lambda (x^2)
         (expt x^2 (/ y 2)))
       (* x x)))))

(define (for-each f . lst)
  (apply map f lst)
  '())

(define (sin x) 0)
(define (cos x)
  (if (zero? x) 1 0))

(define (gcd2 a b)
  (if (zero? a) b
    (if (zero? b) a
      (if (< a b)   (gcd2 a (- b a))
        (gcd2 b (- a b))))))

(define (gcd . lst)
  (foldl1 gcd2 lst))

(define (lcm2 a b)
  (/ (* a b) (gcd2 a b)))

(define (lcm . lst)
  (foldl1 lcm2 lst))

(define (min2 a b)
  (if (< a b) a b))

(define (min . lst)
  (foldl1 min2 lst))

(define (max2 a b)
  (if (> a b) a b))

(define (max . lst)
  (foldl1 max2 lst))


