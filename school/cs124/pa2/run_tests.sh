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
echo "Morton Tiling"
./fast 1 $DIM 1
./fast 1 $DIM 2
./fast 1 $DIM 4
./fast 1 $DIM 8
./fast 1 $DIM 16
./fast 1 $DIM 32
./fast 1 $DIM 64
./fast 1 $DIM 128
./fast 1 $DIM 256

#!/bin/bash
echo "Hybrid"
MODE=5
echo "Plus Minus"
GEN=1
echo "Row Major"
LAY=0
echo "n0=32"
CROSS=32
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
