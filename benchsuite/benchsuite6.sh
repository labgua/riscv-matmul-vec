#!/bin/bash

echo "Benchsuite V6 - perf analysis (tiling vs reordered_tiling [V3])"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <filename>"
    echo "Example: $0 out.txt"
    exit 1
fi

FILENAME="$1"
FILE="report/$FILENAME"

LOGFILE="CURRENT-BENCH.log"
echo "--------------------" >> $LOGFILE

EXE_TILING="./build/riscv64/tiling_v3"
EXE_R_TILING="./build/riscv64/reordered_tiling"

sizes=(256 512 1024 2048 4096)
kernels=(2 4 8 16)
lmuls=(-2)


# EXE_TILING
for size in "${sizes[@]}"; do
    for kernel in "${kernels[@]}"; do
        for lmul in "${lmuls[@]}"; do
            echo "$EXE_TILING) SIZE=$size KERNEL=$kernel LMUL=$lmul"
            echo "$EXE_TILING) SIZE=$size KERNEL=$kernel LMUL=$lmul" >> $LOGFILE
            perf stat -e L1-dcache-loads,L1-dcache-load-misses "$EXE_TILING" SIZE="$size" KERNEL="$kernel" LMUL="$lmul" &>> "$FILE"
        done
    done
done

# EXE_R_TILING
for size in "${sizes[@]}"; do
    for kernel in "${kernels[@]}"; do
        for lmul in "${lmuls[@]}"; do
            echo "$EXE_TILING) SIZE=$size KERNEL=$kernel LMUL=$lmul"
            echo "$EXE_TILING) SIZE=$size KERNEL=$kernel LMUL=$lmul" >> $LOGFILE
            perf stat -e L1-dcache-loads,L1-dcache-load-misses "$EXE_R_TILING" SIZE="$size" KERNEL="$kernel" LMUL="$lmul" &>> "$FILE"
        done
    done
done
