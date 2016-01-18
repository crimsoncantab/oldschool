#!/bin/bash
echo "Hybrid"
MODE=5
echo "Plus Minus"
GEN=1
echo "Row Major"
LAY=0
echo "n0=32"
CROSS=32
./strassen $MODE 250 $GEN $LAY $CROSS 0
./strassen $MODE 275 $GEN $LAY $CROSS 0
./strassen $MODE 300 $GEN $LAY $CROSS 0
./strassen $MODE 325 $GEN $LAY $CROSS 0
./strassen $MODE 350 $GEN $LAY $CROSS 0
./strassen $MODE 375 $GEN $LAY $CROSS 0
./strassen $MODE 400 $GEN $LAY $CROSS 0
./strassen $MODE 425 $GEN $LAY $CROSS 0
./strassen $MODE 450 $GEN $LAY $CROSS 0
./strassen $MODE 475 $GEN $LAY $CROSS 0
./strassen $MODE 500 $GEN $LAY $CROSS 0
./strassen $MODE 525 $GEN $LAY $CROSS 0
