#!/bin/bash

# Controllo argomenti minimi (almeno file ed exe)
if [ $# -lt 2 ]; then
    echo "Usage: $0 <out-filename_path> <executable_path> [SIZE=...] [KERNEL=...] [LMUL=...]"
    echo "Example: $0 report/out.txt ./build/exe SIZE=\"256,512\" KERNEL=\"4,8\" LMUL=\"1,2\""
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

# --- PARSING ARGOMENTI NOMINATI ---
# Iteriamo su tutti gli argomenti a partire dal 3° ($3, $4, ...)
for arg in "${@:3}"; do
    case $arg in
        SIZE=*)
            # Estrae il valore dopo 'SIZE=' e lo converte in array separando per virgola
            IFS=',' read -r -a sizes <<< "${arg#SIZE=}"
            ;;
        KERNEL=*)
            IFS=',' read -r -a kernels <<< "${arg#KERNEL=}"
            ;;
        LMUL=*)
            IFS=',' read -r -a lmuls <<< "${arg#LMUL=}"
            ;;
    esac
done

echo "benchsuite-params.sh (executable: $EXE) (params: SIZE='${sizes[@]}' KERNEL='${kernels[@]}' LMUL='${lmuls[@]}' )"

echo "benchsuite-params.sh (executable: $EXE) (params: SIZE='${sizes[@]}' KERNEL='${kernels[@]}' LMUL='${lmuls[@]}' )" >> "$FILE"



for size in "${sizes[@]}"; do
    for kernel in "${kernels[@]}"; do
        for lmul in "${lmuls[@]}"; do
            echo "$EXE) SIZE=$size KERNEL=$kernel LMUL=$lmul"
            perf stat -e L1-dcache-loads,L1-dcache-load-misses "$EXE" SIZE="$size" KERNEL="$kernel" LMUL="$lmul" &>> "$FILE"
        done
    done
done

