#!/bin/bash
DIM=512
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
