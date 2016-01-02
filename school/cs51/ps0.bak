(define (howmany sym lst)
  (cond ((null? lst) 0)
        ((equal? sym (car lst)) (+ 1 (howmany sym (cdr lst))))
        (else (howmany sym (cdr lst)))
  )
)

(define (sum-squares? a b c)
  (if (number? a) (if (number? b) (if (number? c)
       (if (= (+ (* a a) (* b b)) (* c c)) #t #f)
         #f) #f) #f)
  )

(define (right-triangle? a b c)
  (cond ((sum-squares? a b c) #t)
        ((sum-squares? b c a) #t)
        ((sum-squares? c a b) #t)
        (else #f)
        ))

(define (print-cubes n)
  (display (