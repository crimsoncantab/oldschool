#!/bin/sh

echo a
grep 'Turing,Alan' data.csv;

echo b
grep -E "`grep "Turing,Alan" data.csv | cut -d , -f 6`(,[^,]*){3}$"  data.csv;

echo c
grep -E ',San Francisco(,[^,]*){3}$' data.csv | wc -l;

echo d
grep -E '7th St[^,]*,Chicago(,[^,]*){3}$' data.csv;

echo e
grep '^[^,]*,N' data.csv | cut -d , -f 1-2;

echo f
grep ',23[^,]*,[^,]*$' data.csv | grep -v '^Bradford,Adrian' - | cut -d , -f 1-2,6-8;

echo g
cut -d , -f 6 data.csv | grep -v '^$' | sort | uniq -c | grep -v -E '[[:space:]][1234] ';

echo h
grep -E '([^,]+,){8}.+' data.csv | wc -l;

echo i
grep -E '(8.*){6}' data.csv;

echo j
grep -E '^([^,]*,){2}[^,]*(1..1|2..2|3..3|4..4|5..5|6..6|7..7|8..8|9..9|0..0),' data.csv | grep -E '^([^,]*,){2}[^,]*(11|22|33|44|55|66|77|88|99|00)[^,],' | cut -d , -f1-3;
