#!/bin/bash
echo "Hybrid"
MODE=5
DIM=512
echo "Plus Minus"
GEN=1
echo "Row Major"
LAY=0
./strassen $MODE $DIM $GEN $LAY 1 0
./strassen $MODE $DIM $GEN $LAY 2 0
./strassen $MODE $DIM $GEN $LAY 4 0
./strassen $MODE $DIM $GEN $LAY 8 0
./strassen $MODE $DIM $GEN $LAY 16 0
./strassen $MODE $DIM $GEN $LAY 32 0
./strassen $MODE $DIM $GEN $LAY 64 0
./strassen $MODE $DIM $GEN $LAY 128 0
./strassen $MODE $DIM $GEN $LAY 256 0
echo "Morton"
LAY=2
./strassen $MODE $DIM $GEN $LAY 1 0
./strassen $MODE $DIM $GEN $LAY 2 0
./strassen $MODE $DIM $GEN $LAY 4 0
./strassen $MODE $DIM $GEN $LAY 8 0
./strassen $MODE $DIM $GEN $LAY 16 0
./strassen $MODE $DIM $GEN $LAY 32 0
./strassen $MODE $DIM $GEN $LAY 64 0
./strassen $MODE $DIM $GEN $LAY 128 0
./strassen $MODE $DIM $GEN $LAY 256 0
