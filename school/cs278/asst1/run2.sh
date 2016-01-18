#!/bin/bash

#./bin/trace2 100 > data/part2/100.dat

#./bin/trace2 1000 > data/part2/1000.dat

#./bin/trace2 10000 > data/part2/10000.dat

#./bin/trace2 100000 > data/part2/100000.dat

#./bin/trace2 1000000 > data/part2/1000000.dat

#./bin/trace2 10000000 > data/part2/10000000.dat

#./bin/trace2 100000000 > data/part2/100000000.dat

#echo > data/part2/power
ls -1 data/part2/*.dat | sed s/\.dat//g | xargs -I {} bash -c 'echo "{}:" >> data/part2/power && bin/view 0 1 0 7.06597e-16 < {}.dat > {}.ppm #2>> data/part2/power'

