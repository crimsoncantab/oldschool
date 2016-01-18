#!/bin/bash
CROSS=0
echo "Plus Minus"
GEN=1
echo "Row Major"
LAY=0
echo "Conventional"
MODE=1
./strassen $MODE 16 $GEN $LAY $CROSS 0
./strassen $MODE 32 $GEN $LAY $CROSS 0
./strassen $MODE 64 $GEN $LAY $CROSS 0
./strassen $MODE 128 $GEN $LAY $CROSS 0
./strassen $MODE 256 $GEN $LAY $CROSS 0
./strassen $MODE 512 $GEN $LAY $CROSS 0
./strassen $MODE 1024 $GEN $LAY $CROSS 0
echo "Conventional Cache"
MODE=2
./strassen $MODE 16 $GEN $LAY $CROSS 0
./strassen $MODE 32 $GEN $LAY $CROSS 0
./strassen $MODE 64 $GEN $LAY $CROSS 0
./strassen $MODE 128 $GEN $LAY $CROSS 0
./strassen $MODE 256 $GEN $LAY $CROSS 0
./strassen $MODE 512 $GEN $LAY $CROSS 0
./strassen $MODE 1024 $GEN $LAY $CROSS 0
echo "Strassen"
MODE=3
./strassen $MODE 16 $GEN $LAY $CROSS 0
./strassen $MODE 32 $GEN $LAY $CROSS 0
./strassen $MODE 64 $GEN $LAY $CROSS 0
./strassen $MODE 128 $GEN $LAY $CROSS 0
./strassen $MODE 256 $GEN $LAY $CROSS 0
./strassen $MODE 512 $GEN $LAY $CROSS 0
./strassen $MODE 1024 $GEN $LAY $CROSS 0
echo "Strassen Mem"
MODE=4
./strassen $MODE 16 $GEN $LAY $CROSS 0
./strassen $MODE 32 $GEN $LAY $CROSS 0
./strassen $MODE 64 $GEN $LAY $CROSS 0
./strassen $MODE 128 $GEN $LAY $CROSS 0
./strassen $MODE 256 $GEN $LAY $CROSS 0
./strassen $MODE 512 $GEN $LAY $CROSS 0
./strassen $MODE 1024 $GEN $LAY $CROSS 0
echo "Morton"
LAY=2
echo "Conventional"
MODE=1
./strassen $MODE 16 $GEN $LAY $CROSS 0
./strassen $MODE 32 $GEN $LAY $CROSS 0
./strassen $MODE 64 $GEN $LAY $CROSS 0
./strassen $MODE 128 $GEN $LAY $CROSS 0
./strassen $MODE 256 $GEN $LAY $CROSS 0
./strassen $MODE 512 $GEN $LAY $CROSS 0
./strassen $MODE 1024 $GEN $LAY $CROSS 0
echo "Conventional Cache"
MODE=2
./strassen $MODE 16 $GEN $LAY $CROSS 0
./strassen $MODE 32 $GEN $LAY $CROSS 0
./strassen $MODE 64 $GEN $LAY $CROSS 0
./strassen $MODE 128 $GEN $LAY $CROSS 0
./strassen $MODE 256 $GEN $LAY $CROSS 0
./strassen $MODE 512 $GEN $LAY $CROSS 0
./strassen $MODE 1024 $GEN $LAY $CROSS 0
echo "Strassen"
MODE=3
./strassen $MODE 16 $GEN $LAY $CROSS 0
./strassen $MODE 32 $GEN $LAY $CROSS 0
./strassen $MODE 64 $GEN $LAY $CROSS 0
./strassen $MODE 128 $GEN $LAY $CROSS 0
./strassen $MODE 256 $GEN $LAY $CROSS 0
./strassen $MODE 512 $GEN $LAY $CROSS 0
./strassen $MODE 1024 $GEN $LAY $CROSS 0
echo "Strassen Mem"
MODE=4
./strassen $MODE 16 $GEN $LAY $CROSS 0
./strassen $MODE 32 $GEN $LAY $CROSS 0
./strassen $MODE 64 $GEN $LAY $CROSS 0
./strassen $MODE 128 $GEN $LAY $CROSS 0
./strassen $MODE 256 $GEN $LAY $CROSS 0
./strassen $MODE 512 $GEN $LAY $CROSS 0
./strassen $MODE 1024 $GEN $LAY $CROSS 0
echo "Zero Through Two"
GEN=2
echo "Row Major"
LAY=0
echo "Conventional"
MODE=1
./strassen $MODE 16 $GEN $LAY $CROSS 0
./strassen $MODE 32 $GEN $LAY $CROSS 0
./strassen $MODE 64 $GEN $LAY $CROSS 0
./strassen $MODE 128 $GEN $LAY $CROSS 0
./strassen $MODE 256 $GEN $LAY $CROSS 0
./strassen $MODE 512 $GEN $LAY $CROSS 0
./strassen $MODE 1024 $GEN $LAY $CROSS 0
echo "Conventional Cache"
MODE=2
./strassen $MODE 16 $GEN $LAY $CROSS 0
./strassen $MODE 32 $GEN $LAY $CROSS 0
./strassen $MODE 64 $GEN $LAY $CROSS 0
./strassen $MODE 128 $GEN $LAY $CROSS 0
./strassen $MODE 256 $GEN $LAY $CROSS 0
./strassen $MODE 512 $GEN $LAY $CROSS 0
./strassen $MODE 1024 $GEN $LAY $CROSS 0
echo "Strassen"
MODE=3
./strassen $MODE 16 $GEN $LAY $CROSS 0
./strassen $MODE 32 $GEN $LAY $CROSS 0
./strassen $MODE 64 $GEN $LAY $CROSS 0
./strassen $MODE 128 $GEN $LAY $CROSS 0
./strassen $MODE 256 $GEN $LAY $CROSS 0
./strassen $MODE 512 $GEN $LAY $CROSS 0
./strassen $MODE 1024 $GEN $LAY $CROSS 0
echo "Strassen Mem"
MODE=4
./strassen $MODE 16 $GEN $LAY $CROSS 0
./strassen $MODE 32 $GEN $LAY $CROSS 0
./strassen $MODE 64 $GEN $LAY $CROSS 0
./strassen $MODE 128 $GEN $LAY $CROSS 0
./strassen $MODE 256 $GEN $LAY $CROSS 0
./strassen $MODE 512 $GEN $LAY $CROSS 0
./strassen $MODE 1024 $GEN $LAY $CROSS 0
echo "Morton"
LAY=2
echo "Conventional"
MODE=1
./strassen $MODE 16 $GEN $LAY $CROSS 0
./strassen $MODE 32 $GEN $LAY $CROSS 0
./strassen $MODE 64 $GEN $LAY $CROSS 0
./strassen $MODE 128 $GEN $LAY $CROSS 0
./strassen $MODE 256 $GEN $LAY $CROSS 0
./strassen $MODE 512 $GEN $LAY $CROSS 0
./strassen $MODE 1024 $GEN $LAY $CROSS 0
echo "Conventional Cache"
MODE=2
./strassen $MODE 16 $GEN $LAY $CROSS 0
./strassen $MODE 32 $GEN $LAY $CROSS 0
./strassen $MODE 64 $GEN $LAY $CROSS 0
./strassen $MODE 128 $GEN $LAY $CROSS 0
./strassen $MODE 256 $GEN $LAY $CROSS 0
./strassen $MODE 512 $GEN $LAY $CROSS 0
./strassen $MODE 1024 $GEN $LAY $CROSS 0
echo "Strassen"
MODE=3
./strassen $MODE 16 $GEN $LAY $CROSS 0
./strassen $MODE 32 $GEN $LAY $CROSS 0
./strassen $MODE 64 $GEN $LAY $CROSS 0
./strassen $MODE 128 $GEN $LAY $CROSS 0
./strassen $MODE 256 $GEN $LAY $CROSS 0
./strassen $MODE 512 $GEN $LAY $CROSS 0
./strassen $MODE 1024 $GEN $LAY $CROSS 0
echo "Strassen Mem"
MODE=4
./strassen $MODE 16 $GEN $LAY $CROSS 0
./strassen $MODE 32 $GEN $LAY $CROSS 0
./strassen $MODE 64 $GEN $LAY $CROSS 0
./strassen $MODE 128 $GEN $LAY $CROSS 0
./strassen $MODE 256 $GEN $LAY $CROSS 0
./strassen $MODE 512 $GEN $LAY $CROSS 0
./strassen $MODE 1024 $GEN $LAY $CROSS 0
echo "\nDoing Hybrid Tests"
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
