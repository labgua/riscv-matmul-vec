#!/usr/bin/env python3
"""
Script per estrarre dati dai file di benchmark e generare un CSV strutturato.
Parsa le linee BENCHMARK_RECORD e le statistiche di perf per creare una tabella CSV.
"""

import re
import csv
import sys
from pathlib import Path

def parse_benchmark_file(input_file, output_file):
    """
    Parsa il file di benchmark ed estrae i dati in formato CSV.
    
    Args:
        input_file: Percorso del file di input con output dei benchmark
        output_file: Percorso del file CSV di output
    """
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Trova tutte le linee BENCHMARK_RECORD con le loro posizioni
    bench_pattern = r'> BENCHMARK_RECORD : ([^\n]+)'
    bench_matches = list(re.finditer(bench_pattern, content))
    
    if not bench_matches:
        print("Errore: Nessuna linea BENCHMARK_RECORD trovata nel file")
        return
    
    blocks = []
    
    # Processa ogni BENCHMARK_RECORD e cerca le statistiche di perf associate
    for i, bench_match in enumerate(bench_matches):
        block_data = {}
        
        # Parsa i parametri dalla BENCHMARK_RECORD (chiave=valore)
        params_str = bench_match.group(1).strip()
        for param in params_str.split(','):
            if '=' in param:
                key, value = param.strip().split('=', 1)
                block_data[key.strip()] = value.strip()
        
        # Determina l'area di ricerca per le statistiche di perf
        bench_end = bench_match.end()
        if i < len(bench_matches) - 1:
            next_bench_start = bench_matches[i + 1].start()
            search_area = content[bench_end:next_bench_start]
        else:
            search_area = content[bench_end:]
        
        # Estrae L1-dcache-loads (con gestione separatori migliaia)
        load_match = re.search(r'(\d+(?:,\d+)*)\s+L1-dcache-loads', search_area)
        if load_match:
            loads = int(load_match.group(1).replace(',', ''))
            block_data['l1d-load'] = loads
        
        # Estrae L1-dcache-load-misses e cache miss rate
        miss_match = re.search(r'(\d+(?:,\d+)*)\s+L1-dcache-load-misses\s+#\s+([\d.]+)%', search_area)
        if miss_match:
            misses = int(miss_match.group(1).replace(',', ''))
            rate = float(miss_match.group(2))
            block_data['l1d-misses'] = misses
            block_data['cachemiss-rate'] = rate
        
        if block_data:
            blocks.append(block_data)
    
    if not blocks:
        print("Errore: Nessun blocco di dati valido estratto")
        return
    
    # Determina tutte le colonne possibili dai dati estratti
    all_keys = set()
    for block in blocks:
        all_keys.update(block.keys())
    
    # Ordina le colonne secondo la specifica:
    # version, size, [altri parametri], l1d-load, l1d-misses, cachemiss-rate, time
    ordered_keys = ['version', 'size']
    
    # Parametri aggiuntivi nell'ordine specificato (solo quelli presenti)
    additional_params = ['kernel', 'lmul', 'unroll']
    for param in additional_params:
        if param in all_keys:
            ordered_keys.append(param)
    
    # Colonne fisse delle statistiche
    ordered_keys.extend(['l1d-load', 'l1d-misses', 'cachemiss-rate', 'time'])
    
    # Scrive il file CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(blocks)
    
    print(f"✓ File CSV generato con successo: {output_file}")
    print(f"  {len(blocks)} esecuzioni estratte")
    print(f"  Colonne: {', '.join(ordered_keys)}")


def main():
    """Funzione principale con gestione degli argomenti"""
    if len(sys.argv) != 3:
        print("Utilizzo: python parse_benchmarks.py <file_input> <file_output.csv>")
        print("\nEsempio: python parse_benchmarks.py benchmark_output.txt results.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"Errore: File di input non trovato: {input_file}")
        sys.exit(1)
    
    parse_benchmark_file(input_file, output_file)


if __name__ == '__main__':
    main()