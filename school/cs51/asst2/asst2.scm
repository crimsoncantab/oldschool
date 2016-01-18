;; CS51 Assignment 2
;; 
;; File:      asst2.scm
;; Author:    Alex Yang <alexyang@fas.harvard.edu>
;; Partner:   Loren McGinnis <mcginn@fas.harvard.edu>
;; 
;; Collaboration is required on this assignment.
;; By submitting this assignment, you certify that you have
;; followed the CS51 collaboration policy.  If you are unclear
;; on the CS51 collaboration policy, please ask your TF or
;; the head TF.


;;;;;
;;;;; Exercise 1
;;;;;



; The function (filter pred? lst) takes a predicate and a
; list, and returns a new list containing the elements of
; lst that satisfy pred?.  For example:

(define (filter pred? lst)
  (cond
    ((null? lst)       '())
    ((pred? (car lst)) (cons (car lst) (filter pred? (cdr lst))))
    (else              (filter pred? (cdr lst)))))


;;;;;
;;;;; Exercise 2
;;;;;

; Write the following functions using filter and lambda:

;;; (a)
;
; (filter-<-3 lst) returns (a list containing) the elements
; of lst that are less than 3.

(define (filter-<-3 lst)
   (filter (lambda (x) (< x 3)) lst))


;;; (b)
;
; (filter-< n lst) returns the elements of lst that are less
; than n.

(define (filter-< n lst)
  (filter (lambda (x) (< x n)) lst))

;;; (c)
;
; (filter-or p1? p2? lst) returns the elements of lst that
; satisfy either p1? or p2?.

(define (filter-or p1? p2? lst)
  (filter (lambda (x) (or (p1? x) (p2? x))) lst))


;;;;;
;;;;; Exercise 3
;;;;;


;;; (a)
;
; Write filter in terms of reduce.
; 
; Here is a definition of reduce:
(define (reduce func base lst)
   (if (null? lst)
       base
       (func (car lst) (reduce func base (cdr lst)))))

(define (my-filter pred? lst)
   (reduce (lambda (x y) (if (pred? x) (cons x y) y)) '() lst))

;;; (b)
;
(define (all pred? lst)
  (reduce (lambda (x y) (if (pred? x) (and #t y) #f)) #t lst))

;;; (c)
;
(define (my-max lst)
  (reduce (lambda (x y) (max x y)) (car lst) lst))


; Since we're defining grammars in terms of dictionaries,
; we need a dictionary implementation.  You may use your
; own or the one we've provided:

(define dict-new dict51-new)
(define dict-extend dict51-extend)
(define dict-lookup dict51-lookup)
(define dict-keys dict51-keys)


;;;;;
;;;;; Exercise 4
;;;;;


;;; (a)
;
; Write make-grammar.  You should use reduce in your answer.

(define (make-grammar rules)
  (reduce (lambda (x y) (dict-extend y (car x) (cdr x))) (dict-new) rules))

;;; (b)
;
; Define sentence-grammar to be the Scheme representation of
; the <sentence> grammar above, using make-grammar.  Think
; carefully about how this data structure is constructed,
; because your code will have to traverse it correctly.

(define sentence-grammar
  (make-grammar
     '((sentence ((noun-phrase verb-phrase |.|) (who verb-phrase ?)))
       (noun-phrase ((article modified-noun) (proper-noun)))
       (article ((a) (the)))
       (modified-noun ((noun) (adjective modified-noun)))
       (noun ((seal) (walrus) (dolphin)))
       (adjective ((wet) (green) (tuna-safe)))
       (proper-noun ((pedro martinez) (wally) (flipper)))
       (verb-phrase ((transitive-verb noun-phrase) (intransitive-verb)))
       (transitive-verb ((ate) (watched) (swam around)))
       (intransitive-verb ((swam) (lolled about) (ate))))))


;;;;;
;;;;; Exercise 5
;;;;;

; Write generate.

; returns a random element of the given list
(define (pick-random elements)
  (list-ref elements (random (length elements))))

(define (generate grammar lhs)

  ; base case returns terminal value
  (if (null? (dict-lookup grammar lhs)) (list lhs)
        
        ; recursively traverses the grammar structure and appends terminal values
        (reduce (lambda (x y) (append (generate grammar x) y)) '() 
        
        ; selects an option from a list of alternatives
        (pick-random (car (dict-lookup grammar lhs))))))

;;;;;
;;;;; Exercise 6
;;;;;

; Write your own grammar using make-grammar.  Call it
; my-grammar, and make it interesting.

(define my-grammar
  (make-grammar
   '((sentence ((statement |.|) (question ?) (statement conj statement |.|)))
     (statement ((noun-phrase predicate adverb)))
     (question ((where is noun-phrase)))
     (noun-phrase ((article modified-noun) (proper-noun)))
     (article ((a) (the) (his) (her)))
     (modified-noun ((noun) (adjective modified-noun)))
     (noun ((potato) (flip-flop) (vigor) (accordion) (chestnut) (salad) (orchestra)))
     (adjective ((honey dijon) (sketchy) (ambivalent) (frivolous) (muffled)))
     (proper-noun ((antimony) (disney world) (alex) (loren) (CS51)))
     (predicate ((verb-phrase) (verb-phrase prep-phrase)))
     (prep-phrase ((prep noun-phrase)))
     (conj ((|and|) (but) (|or|)))
     (prep ((in) (on) (behind) (with) (underneath) (about) (with)))
     (verb-phrase ((transitive-verb noun-phrase) (intransitive-verb)))
     (adverb ((surreptitously) (slowly) (yesterday) (quickly) (jovially)))
     (transitive-verb ((defenestrated) (smelled) (pondered) (pwned) (shot) (knocked up)
                       (overturned)))
     (intransitive-verb ((tip-toed) (fell in love) (danced))))))
     

; Show us what it can do by putting your best few generated
; results in a comment.  If you think it would work better
; in a separate file, tell us where to look.
;
; (alex knocked up a salad surreptitously |.|)
; (disney world knocked up loren with vigor |.|)


;;;;;
;;;;; Exercise 7
;;;;;

; If you'd like to do the next exercise first and write the
; bigrams functions in terms of n-grams functions, you may.
; Beware---it might be more difficult, and you could lose
; points on both.


;;; (a)
;
; Write a function (read-bigrams filename) that returns a
; list of bigrams taken from the corpus in filename.

; returns bigrams from a given list
(define (list-bigrams lst)
  (if (null? (cdr lst))
    '()
    ; recursively appends a bigram to the front of a list
    (cons (list (car lst) (cadr lst)) (list-bigrams (cdr lst)))))

(define (read-bigrams filename)
  (list-bigrams (read-tokens filename)))


;;; (b)
;
; Write a function (babble-bigrams bigrams len) that returns
; a babble from bigrams as a list.  It should be at least
; len symbols long and end in eop.

; returns a bigram beginning with eop
(define (pick-first-bigram bigrams)
  (pick-next-bigram bigrams (list '() 'eop)))

; returns a random bigram that matches with given bigram
(define (pick-next-bigram bigrams bigram)
  (pick-random (filter (lambda (x) (equal? (car x) (cadr bigram))) bigrams)))

; recursively appends bigrams that match up
(define (append-bigrams bigrams bigram len)

  ; stops when desired length has been reached and next bigram has eop
  (if (and (<= len 2) (equal? (cdr bigram) '(eop))) bigram

      ; recursivesly adds the next bigram, decrementing len by 1
      (cons (car bigram) (append-bigrams bigrams (pick-next-bigram bigrams bigram)
                                         (- len 1)))))

(define (babble-bigrams bigrams len)
  (append-bigrams bigrams (pick-first-bigram bigrams) len))


;;;;;
;;;;; Exercise 8
;;;;;


;;; (a)
;
; Write a function (read-ngrams n filename) that returns a
; list of n-grams taken from the corpus in filename.

; returns first n elements of given list
(define (front n lst)
  (if (= n 0) '()
      (append (list (car lst)) (front (- n 1) (cdr lst)))))

; appends n elements from beginning of list to end
(define (wrap-list n lst)
  (append lst (front n lst)))

; returns list of ngrams from given list
(define (list-ngrams n lst)
  (if (= (length lst) n)
    '()
    (cons (front n lst) (list-ngrams n (cdr lst)))))

(define (read-ngrams n filename)
  (list-ngrams n (wrap-list n (read-tokens filename))))

;;; (b)
;
; Write (babble-ngrams n ngrams len) to return a babble from
; ngrams as a list.  It should be at least len symbols long
; and end in eop.

; returns an ngram beginning with eop
(define (pick-first-ngram n ngrams)
  (pick-random (filter (lambda (x) (equal? (car x) 'eop)) ngrams)))

; returns a random ngram that matches the given ngram
(define (pick-next-ngram n ngrams ngram)
  (pick-random (filter (lambda (x) (equal? (front (- n 1) x) (cdr ngram))) ngrams)))

; recursively appends ngrams that match up
(define (append-ngrams n ngrams ngram len)
  
  ; stops when desired length has been reached and next ngram ends with eop
  (if (and (<= len n) (equal? (list-ref ngram (- n 1)) 'eop)) ngram
      
      ; recursivesly adds the next ngram, decrementing len by 1
      (cons (car ngram) (append-ngrams n ngrams (pick-next-ngram n ngrams ngram) 
                                       (- len 1)))))

(define (babble-ngrams n ngrams len)
  (append-ngrams n ngrams (pick-first-ngram n ngrams) len))

; We've provided a corpus, t-and-c.txt for you test your
; babbler.  After you have your babbler working, find a
; large corpus of text (10,000 words or more) that generates
; amusing results.  Save the input in corpus.txt and your
; most amusing result(s) in the file output.txt.


;;;;;
;;;;; WHEN YOU ARE FINISHED
;;;;;

; About how long did this take you to complete?  How
; confident are you in your code?  In addition, please
; indicate how long you believe your partner worked on the
; assignment as well as how long you worked on it; this
; is the only part of the assignment that should differ
; between your submission and your partner's submission.
; Your answer will not affect your grade.  Type your answer
; in a comment below.
;
; This assignment took me about 7 hours to complete.  I am quite
; confident about my code.  My partner and I worked together for
; essentially all of the problem set, so he also spent about
; 7 hours on it.

