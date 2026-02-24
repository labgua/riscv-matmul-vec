#!/bin/bash

echo "Benchsuite V7 - perf analysis (tiling vs reordered_tiling with UNROLLING [V3]) - 100 execution"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <filename>"
    echo "Example: $0 out.txt"
    exit 1
fi

FILENAME="$1"
FILE="report/$FILENAME"

LOGFILE="CURRENT-BENCH.log"
echo "--------------------" >> $LOGFILE

exes_r_tiling_unrolling=(
    "./build/riscv64/reordered_tiling_unrolling2"
    "./build/riscv64/reordered_tiling_unrolling4"
    "./build/riscv64/reordered_tiling_unrolling8"
    "./build/riscv64/reordered_tiling_unrolling16"
)

configuration=(
    "KERNEL=2 LMUL=2"
    "KERNEL=4 LMUL=2"
    "KERNEL=4 LMUL=4"
    "KERNEL=4 LMUL=8"
    "KERNEL=8 LMUL=8"
)

sizes=(256 512 1024 2048 4096)

for config in "${configuration[@]}"; do
    echo "CONFIG> $config"
    for size in "${sizes[@]}"; do
        for exe_unrolled in "${exes_r_tiling_unrolling[@]}"; do
            echo "perf stat -e L1-dcache-loads,L1-dcache-load-misses $exe_unrolled SIZE=$size $config"
            echo "perf stat -e L1-dcache-loads,L1-dcache-load-misses $exe_unrolled SIZE=$size $config" >> $LOGFILE
            perf stat -e L1-dcache-loads,L1-dcache-load-misses "$exe_unrolled" SIZE="$size" $config &>> "$FILE"
        done
        echo "---------------------------------------------------"
        echo "---------------------------------------------------" >> $LOGFILE
    done
done
