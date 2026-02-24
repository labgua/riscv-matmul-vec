#!/bin/bash

echo "Benchmark - execute all measures"

PREFIX="bench$(date +%d%m%y)"

# baseline
echo "> baseline"
mkdir report/baseline
./benchsuite/benchsuite-size.sh report/baseline/$PREFIX-baseline.txt ./build/risc64/baseline

# autovect
echo "> autovect"
mkdir report/autovect
./benchsuite/benchsuite-size.sh report/autovect/$PREFIX-autovect.txt ./build/risc64/autovect

# tiling_v1
echo "> tiling_v1"
mkdir report/tiling_v1
./benchsuite/benchsuite-params.sh \
    report/tiling_v1/$PREFIX-tiling_v1.txt \
    ./build/risc64/tiling \
    SIZE="256,512,1024,2048,4096" \
    KERNEL="2,4,8" \
    LMUL="1"

# tiling_v2
echo "> tiling_v2"
mkdir report/tiling_v2
./benchsuite/benchsuite-params.sh \
    report/tiling_v2/$PREFIX-tiling_v2.txt \
    ./build/risc64/tiling_v2 \
    SIZE="256,512,1024,2048,4096" \
    KERNEL="2,4,8,16" \
    LMUL="1"

# tiling_v3
echo "> tiling_v3"
mkdir report/tiling_v3
./benchsuite/benchsuite-params.sh \
    report/tiling_v3/$PREFIX-tiling_v3.txt \
    ./build/risc64/tiling_v3 \
    SIZE="256,512,1024,2048,4096" \
    KERNEL="2,4,8,16" \
    LMUL="-2,1,2,4,8"

# reordered_tiling
echo "> reordered_tiling"
mkdir report/reordered_tiling
./benchsuite/benchsuite-params.sh \
    report/reordered_tiling/$PREFIX-reordered_tiling.txt \
    ./build/risc64/reordered_tiling \
    SIZE="256,512,1024,2048,4096" \
    KERNEL="2,4,8,16" \
    LMUL="-2,1,2,4,8"

# reordered_tiling_unrolling
echo "> reordered_tiling_unrolling"
mkdir report/reordered_tiling_unrolling
./benchsuite/benchsuite-custom.sh