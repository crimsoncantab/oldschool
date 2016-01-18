nums="6 8 10 12 16 20 24 32 40 100"

for number in $nums
do
echo "$number"
./sum_file < $number.out
done

