#!/bin/bash

EXPECTED_ARGS=1
E_BADARGS=65

if [ $# -ne $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` {arg}"
  exit $E_BADARGS
fi

make clean
make MODE=prof mass_spring shallow_water

echo '=='$1'==
Unoptimized' >> PERF.txt

echo '
./mass_spring grid3* 10' >> PERF.txt
./mass_spring grid3* 10 >> PERF.txt
echo '
./shallow_water pond4* 0 10' >> PERF.txt
./shallow_water pond4* 0 10 >> PERF.txt
echo '' >> PERF.txt

valgrind --callgrind-out-file=perf/ms_$1.out --tool=callgrind ./mass_spring grid3* 10
valgrind --callgrind-out-file=perf/sw_$1.out --tool=callgrind ./shallow_water pond4* 0 10

make clean
make MODE=fast mass_spring shallow_water

echo 'Optimized' >> PERF.txt

echo '
./mass_spring grid3* 10' >> PERF.txt
./mass_spring grid3* 10 >> PERF.txt
echo '
./shallow_water pond4* 0 10' >> PERF.txt
./shallow_water pond4* 0 10 >> PERF.txt
echo '' >> PERF.txt
