#!/bin/bash
echo "Hybrid"
MODE=5
echo "Plus Minus"
GEN=1
echo "Row Major"
LAY=0
echo "n0=32"
CROSS=8
./fast 1 250 $CROSS
./fast 1 275 $CROSS
./fast 1 300 $CROSS
./fast 1 325 $CROSS
./fast 1 350 $CROSS
./fast 1 375 $CROSS
./fast 1 400 $CROSS
./fast 1 425 $CROSS
./fast 1 450 $CROSS
./fast 1 475 $CROSS
./fast 1 500 $CROSS
./fast 1 525 $CROSS
