#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <out-filename_path> <executable_path>"
    echo "Example: $0 report/baseline/bench-out.txt ./build/riscv64/baseline"
    exit 1
fi

FILE="$1"
EXE="$2"

# Verifica che l'eseguibile esista
if [ ! -x "$EXE" ]; then
    echo "Error: executable '$EXE' not found or not executable"
    exit 1
fi

# Crea la directory report se non esiste
mkdir -p report


echo "benchsuite-size.sh (executable: $EXE)"

echo "benchsuite-size.sh (executable: $EXE)" >> "$FILE"

sizes=(256 512 1024 2048 4096)


# Modalità baseline: solo size, nessun kernel
for size in "${sizes[@]}"; do
    echo "$EXE) SIZE=$size"
    perf stat -e L1-dcache-loads,L1-dcache-load-misses "$EXE" SIZE=$size &>> "$FILE"
done

echo "Done."
