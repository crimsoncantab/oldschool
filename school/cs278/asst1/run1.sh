#!/bin/bash

#./bin/trace1 10 > data/part1/10.dat

#./bin/trace1 100 > data/part1/100.dat

#./bin/trace1 1000 > data/part1/1000.dat

#./bin/trace1 10000 > data/part1/10000.dat

#./bin/trace1 100000 > data/part1/100000.dat

#takes too long
#./bin/trace1 1000000 > data/part1/1000000.dat

#echo > data/part1/power
ls -1 data/part1/*.dat | sed s/\.dat//g | xargs -I {} bash -c 'echo "{}:" >> data/part1/power && bin/view 1 0 0 7.24152e-16 < {}.dat > {}.ppm #2>> data/part1/power'

