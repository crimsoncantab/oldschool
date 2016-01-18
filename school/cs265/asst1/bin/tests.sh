#!/bin/sh
nums="6 8 10 12 16 20 24 32 40 100"

for number in $nums
do
echo "Doing strings of length $number"
./string_picker $1 $2 $number 125 | xargs -I xxxx simplequery $1 xxxx > $number.out
done

sh ./count.sh > $1.count.out
