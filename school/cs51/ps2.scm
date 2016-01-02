
(define (filter pred? lst)
  (cond
    ((null? lst)       '())
    ((pred? (car lst)) (cons (car lst) (filter pred? (cdr lst))))
    (else              (filter pred? (cdr lst)))))


(define (filter-<-3 lst)
  (filter (lambda (x) (< x 3)) lst))

(define (filter-< n lst)
  (filter (lambda (x) (< x n)) lst))

(define (filter-or p1? p2? lst)
  (filter (lambda (x) (or (p1? x) (p2? x))) lst))

(define (reduce func base lst)
   (if (null? lst)
       base
       (func (car lst) (reduce func base (cdr lst)))))

(define (my-filter pred? lst)
   (reduce (lambda (x y) (if (pred? x) (cons x y) y))  '() lst))

(define (all pred? lst)
   (reduce (lambda (x y) (if (pred? x) (and #t y) #f)) #t lst))

(define (my-max lst)
   (reduce (lambda (x y) (max x y)) (car lst) lst))

(define list1 (list 1 3 8 7))
