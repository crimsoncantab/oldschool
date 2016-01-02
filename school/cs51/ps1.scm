(define (mystery x)
  (cond
    ((null? x)        x)
    ((null? (car x))  (mystery (cdr x)))
    ((pair? (car x))  (append (mystery (car x)) (mystery (cdr x))))
    (else             (cons (car x) (mystery (cdr x))))))

(define (restrict enzyme dna)
  (restrict-position-helper enzyme dna 0))

(define (restrict-position-helper enzyme dna position)
  (cond
    ((null? dna) '())
    ((restrict-mystery-helper? enzyme dna)
      (cons position
            (restrict-position-helper enzyme (cdr dna) (+ 1 position))))
    (else (restrict-position-helper enzyme (cdr dna) (+ 1 position)))))

(define (restrict-mystery-helper? enzyme dna)
  (cond
    ((and (null? dna) (not (null? enzyme))) #f)
    ((null? enzyme) #t)
    ((equal? (car enzyme) (car dna))
       (restrict-mystery-helper? (cdr enzyme) (cdr dna)))
    (else #f)))

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

(define (tree-make-empty) '())
(define (tree-empty? tree) (null? tree))
(define (tree-make-node obj left right) (list obj left right))
(define (tree-obj tree) (car tree))
(define (tree-left tree) (cadr tree))
(define (tree-right tree) (caddr tree))

(define (dicttree-make-node key value left right)
  (tree-make-node (cons key value) left right))
(define (dicttree-node-key dt) (car (tree-obj dt)))
(define (dicttree-node-value dt) (cdr (tree-obj dt)))
(define (symbol<? a b)
  (string<? (symbol->string a) (symbol->string b)))

(define (dicttree-new) (tree-make-empty))

(define (dicttree-lookup dt key)
  (cond
    ((tree-empty? dt)            '())
    ((equal? key (dicttree-node-key dt)) (dicttree-node-value dt))
    ((symbol<? key (dicttree-node-key dt))
      (dicttree-lookup (tree-left dt) key))
    (else
      (dicttree-lookup (tree-right dt) key))))

(define (dicttree-keys dt)
  (mystery
    (cond
      ((tree-empty? dt) '())
      (else (cons (dicttree-keys (tree-left dt))
                  (cons (dicttree-node-key dt)
                    (dicttree-keys (tree-right dt))))))))

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