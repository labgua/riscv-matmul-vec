#!/usr/bin/env python3
import csv
import sys
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib import rcParams
import os

def main():
    # Imposta font leggibili
    rcParams['font.family'] = 'DejaVu Sans'
    rcParams['font.size'] = 9

    # Configuriamo gli argomenti a riga di comando
    parser = argparse.ArgumentParser(description='Benchmark plotting tool')
    parser.add_argument('csv_file', help='File CSV da elaborare')
    parser.add_argument('--mode', choices=['auto', 'single', 'compare'], default='auto',
                        help='Modalità di elaborazione: auto (default), single (una sola versione), compare (confronto tra versioni)')
    parser.add_argument('--no-global', action='store_true', 
                        help="Non generare il grafico globale con tutte le configurazioni")
    parser.add_argument('--no-version', action='store_true',
                        help="Non generare i grafici per versione")
    parser.add_argument('--no-size', action='store_true',
                        help="Non generare i grafici a barre per ogni size (tutte le versioni)")
    parser.add_argument('--no-version-size', action='store_true',
                        help="Non generare i grafici a barre per (versione × size)")
    parser.add_argument('--compare-files', nargs='+', 
                        help='File aggiuntivi per il confronto tra diversi benchmark')
    parser.add_argument('--output-dir', default='.',
                        help='Directory di output per i grafici generati')
    
    args = parser.parse_args()
    
    # Crea la directory di output se non esiste
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Leggiamo il/i file CSV
    all_runs = []
    files_to_process = [args.csv_file] + (args.compare_files or [])
    
    for filename in files_to_process:
        try:
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)
                fieldnames = [name.strip() for name in reader.fieldnames]
                
                # Determina i nomi delle colonne (case-insensitive e con possibili varianti)
                version_col = next((col for col in fieldnames 
                                   if col.lower() in ['nome versione', 'version']), None)
                size_col = next((col for col in fieldnames 
                                if col.lower() == 'input size'), None)
                kernel_col = next((col for col in fieldnames 
                                  if col.lower() == 'kernel'), None)
                lmul_col = next((col for col in fieldnames 
                                if col.lower() == 'lmul'), None)
                unroll_col = next((col for col in fieldnames 
                                  if col.lower() == 'unroll'), None)
                loads_col = next((col for col in fieldnames 
                                 if 'l1-dcache-loads' in col.lower()), None)
                misses_col = next((col for col in fieldnames 
                                  if 'l1-dcache-load-misses' in col.lower()), None)
                miss_rate_col = next((col for col in fieldnames 
                                     if 'cache miss rate' in col.lower()), None)
                time_col = next((col for col in fieldnames 
                                if 'tempo di esecuzione' in col.lower() 
                                or 'exec time' in col.lower()), None)

                # Verifica che le colonne necessarie siano presenti
                required_cols = [version_col, size_col, loads_col, misses_col, time_col]
                if not all(required_cols):
                    print("Errore: colonne necessarie non trovate nel CSV")
                    print(f"Versione: {version_col is not None}, Size: {size_col is not None}, "
                          f"Loads: {loads_col is not None}, Misses: {misses_col is not None}, "
                          f"Time: {time_col is not None}")
                    sys.exit(1)

                for row in reader:
                    # Estrai e pulisci i dati
                    version = row[version_col].strip()
                    size = int(row[size_col])
                    
                    # Kernel: se presente, altrimenti "N/A"
                    kernel = row[kernel_col].strip() if kernel_col and row[kernel_col].strip() else "N/A"
                    
                    # LMUL: se presente, altrimenti "1" (valore di default)
                    lmul = row[lmul_col].strip() if lmul_col and row[lmul_col].strip() else "1"
                    
                    # UNROLL: se presente, altrimenti "1" (valore di default)
                    unroll = row[unroll_col].strip() if unroll_col and row[unroll_col].strip() else "1"
                    
                    # Loads e misses: rimuovi le virgole (se presenti) prima di convertire in int
                    loads = int(row[loads_col].replace(',', '').strip())
                    misses = int(row[misses_col].replace(',', '').strip())
                    
                    # Miss rate: se presente, altrimenti calcolalo
                    if miss_rate_col and row[miss_rate_col].strip():
                        miss_rate = float(row[miss_rate_col].strip())
                    else:
                        miss_rate = (misses / loads * 100) if loads else 0.0
                    
                    # Tempo di esecuzione
                    time = float(row[time_col].strip())
                    
                    all_runs.append({
                        "version": version,
                        "size": size,
                        "kernel": kernel,
                        "lmul": lmul,
                        "unroll": unroll,
                        "time": time,
                        "loads": loads,
                        "misses": misses,
                        "miss_rate": miss_rate,
                        "source_file": filename  # Per tracciare il file sorgente
                    })
        except Exception as e:
            print(f"Errore durante l'elaborazione del file {filename}: {str(e)}")
            sys.exit(1)
    
    if not all_runs:
        print("Nessun dato trovato nei file CSV!")
        sys.exit(1)
    
    # Controlliamo quante versioni uniche ci sono
    unique_versions = set(run["version"] for run in all_runs)
    num_versions = len(unique_versions)
    
    # Determiniamo automaticamente la modalità se non specificata
    if args.mode == 'auto':
        if num_versions == 1:
            args.mode = 'single'
            print(f"Rilevata una sola versione ({list(unique_versions)[0]}). Modalità 'single' attivata automaticamente.")
        else:
            args.mode = 'compare'
            print(f"Rilevate {num_versions} versioni. Modalità 'compare' attivata automaticamente.")
    
    # Impostiamo le opzioni predefinite in base alla modalità
    if args.mode == 'single':
        # In modalità singola, disattiviamo alcune opzioni per default
        if not args.no_global:
            args.no_global = True
        if not args.no_size:
            args.no_size = True
    # In modalità confronto, manteniamo le opzioni come specificate
    
    # ==================== DEFINIZIONE PALETTE ESTESA ====================
    # 60 colori distinti da 3 colormap qualitative
    color_cycle = (
        [plt.cm.tab20(i) for i in range(20)] +
        [plt.cm.tab20b(i) for i in range(20)] +
        [plt.cm.tab20c(i) for i in range(20)]
    )
    # 4 stili linea distinti
    linestyle_cycle = ['-', '--', '-.', ':']
    # 12 marcatori distinti
    marker_cycle = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']

    # ==================== FUNZIONE PER ETICHETTA VERSIONE COMPATTA ====================
    def get_short_label(version, kernel, lmul, unroll="1"):
        """
        Genera etichetta compatta che mostra il nome effettivo della versione con i parametri
        
        Questa funzione gestisce correttamente:
        1. Il nome effettivo della versione (non solo "tiling" o "reordered tiling")
        2. LMUL frazionali (es. -2 → 1/2)
        3. Parametri opzionali (solo se diversi dai valori di default)
        """
        # Se la versione non ha informazioni specifiche sui parametri, mostriamo solo il nome
        if kernel == "N/A" and lmul == "1" and unroll == "1":
            return version
        
        # Altrimenti, mostriamo il nome della versione con i parametri rilevanti
        params = []
        if kernel != "N/A" and kernel != "0":
            params.append(f"k={kernel}")
        if lmul != "1":
            # Converti LMUL frazionale in rappresentazione leggibile
            try:
                lmul_val = int(lmul)
                if lmul_val < 0:
                    frac_map = {-2: "1/2", -4: "1/4", -8: "1/8"}
                    lmul_display = frac_map.get(lmul_val, f"1/{abs(lmul_val)}")
                    params.append(f"LMUL={lmul_display}")
                else:
                    params.append(f"LMUL={lmul}")
            except:
                params.append(f"LMUL={lmul}")
        if unroll != "1":
            params.append(f"UNROLL={unroll}")
        
        # Costruisci l'etichetta
        if params:
            return f"{version} ({', '.join(params)})"
        else:
            return version

    # ==================== GRAFICO GLOBALE CON LEGENDA ====================
    if not args.no_global and args.mode == 'compare':
        series_all = defaultdict(list)
        for run in all_runs:
            key = (run["version"], run["kernel"], run["lmul"], run["unroll"])
            series_all[key].append((run["size"], run["time"]))
        sorted_keys = sorted(series_all.keys())
        total_series = len(sorted_keys)
        plt.figure(figsize=(15, 9))
        for idx, key in enumerate(sorted_keys):
            version, kernel, lmul, unroll = key
            points = sorted(series_all[key], key=lambda x: x[0])
            sizes, times = zip(*points)
            # Assegna stile unico combinando colore, linea e marcatore
            color = color_cycle[idx % len(color_cycle)]
            linestyle = linestyle_cycle[idx % len(linestyle_cycle)]
            marker = marker_cycle[idx % len(marker_cycle)]

            label = get_short_label(version, kernel, lmul, unroll)
            plt.plot(sizes, times,
                     color=color,
                     linestyle=linestyle,
                     marker=marker,
                     linewidth=2.0,
                     markersize=6,
                     alpha=0.85,
                     label=label)
        # Configura legenda multi-colonna per evitare sovrapposizioni
        ncol = 4 if total_series > 30 else 3 if total_series > 20 else 2 if total_series > 10 else 1
        plt.legend(loc='upper left', fontsize=7, ncol=ncol, framealpha=0.9,
                   handletextpad=0.5, columnspacing=1.0, labelspacing=0.3)
        plt.xlabel("Input size", fontsize=13, fontweight="bold")
        plt.ylabel("Execution time (s)", fontsize=13, fontweight="bold")
        plt.title("Execution time vs Input size (all configurations)", fontsize=15, fontweight="bold", pad=15)
        plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "benchmark_all_versions.png"), dpi=200, bbox_inches="tight")

    # ==================== GRAFICI PER VERSIONE (fissata) ====================
    if not args.no_version:
        version_groups = defaultdict(list)
        for run in all_runs:
            version_groups[run["version"]].append(run)
        
        for version_name, runs_in_version in sorted(version_groups.items()):
            plt.figure(figsize=(12, 7))
            config_series = defaultdict(list)
            for run in runs_in_version:
                key = (run["kernel"], run["lmul"], run["unroll"])
                config_series[key].append((run["size"], run["time"]))

            sorted_configs = sorted(config_series.keys())
            total_local = len(sorted_configs)

            for idx, config in enumerate(sorted_configs):
                kernel, lmul, unroll = config
                points = sorted(config_series[config], key=lambda x: x[0])
                sizes, times = zip(*points)
                
                color = color_cycle[idx % len(color_cycle)]
                linestyle = linestyle_cycle[idx % len(linestyle_cycle)]
                marker = marker_cycle[idx % len(marker_cycle)]
                
                # Formatta LMUL frazionale anche nelle legende per-versione
                try:
                    lmul_val = int(lmul)
                    if lmul_val < 0:
                        frac_map = {-2: "1/2", -4: "1/4", -8: "1/8"}
                        lmul_display = frac_map.get(lmul_val, f"1/{abs(lmul_val)}")
                    else:
                        lmul_display = str(lmul_val)
                except:
                    lmul_display = lmul
                
                unroll_suffix = f" UNROLL={unroll}" if unroll != "1" else ""
                label = f"k={kernel} LMUL={lmul_display}{unroll_suffix}"
                plt.plot(sizes, times,
                         color=color,
                         linestyle=linestyle,
                         marker=marker,
                         linewidth=2.3,
                         markersize=7,
                         alpha=0.9,
                         label=label)

            # Legenda con 2 colonne per versioni con molte configurazioni
            ncol_local = 2 if total_local > 8 else 1
            plt.legend(loc='upper left', fontsize=9, ncol=ncol_local, framealpha=0.95,
                       handletextpad=0.5, columnspacing=1.2, labelspacing=0.4)
            plt.xlabel("Input size", fontsize=12, fontweight="bold")
            plt.ylabel("Execution time (s)", fontsize=12, fontweight="bold")
            plt.title(f"Execution time vs Input size\n({get_short_label(version_name, '*', '*', '1').split(' (')[0]})", 
                      fontsize=14, fontweight="bold", pad=12)
            plt.grid(True, alpha=0.35, linestyle="--", linewidth=0.8)
            plt.tight_layout()
            safe_version = version_name.replace(" ", "_").replace("/", "_").replace(".", "_")
            plt.savefig(os.path.join(args.output_dir, f"benchmark_{safe_version}.png"), dpi=200, bbox_inches="tight")

    # ==================== GRAFICI A BARRE PER OGNI SIZE (LABELS ACCORCIATE) ====================
    if not args.no_size and args.mode == 'compare':
        unique_sizes = sorted(set(run["size"] for run in all_runs))
        for size in unique_sizes:
            runs_at_size = [run for run in all_runs if run["size"] == size]
            if not runs_at_size:
                continue
                
            # Ordinamento per versione (non per configurazione specifica)
            runs_at_size.sort(key=lambda r: (
                r["version"],
                int(r["kernel"]) if r["kernel"].isdigit() else 0,
                int(r["lmul"]) if r["lmul"].lstrip('-').isdigit() else 0,  # gestisce anche "-2"
                int(r["unroll"]) if r["unroll"].isdigit() else 0
            ))

            # Etichette compatte per le barre (usa get_short_label che mostra il nome effettivo della versione)
            labels = [
                get_short_label(run['version'], run['kernel'], run['lmul'], run['unroll'])
                for run in runs_at_size
            ]

            loads = [run["loads"] for run in runs_at_size]
            misses = [run["misses"] for run in runs_at_size]

            x = np.arange(len(labels))
            bar_width = 0.35

            fig, ax = plt.subplots(figsize=(15, 9.5))

            bars1 = ax.bar(x - bar_width/2, loads, bar_width, 
                          label="L1 loads", color="#27ae60", alpha=0.85, 
                          edgecolor='black', linewidth=0.6)
            bars2 = ax.bar(x + bar_width/2, misses, bar_width, 
                          label="L1 misses", color="#c0392b", alpha=0.85, 
                          edgecolor='black', linewidth=0.6)

            # Aggiungi percentuale miss rate sopra le barre misses
            for i, (load, miss) in enumerate(zip(loads, misses)):
                miss_rate = (miss / load * 100) if load else 0
                ax.text(x[i] + bar_width/2, miss + max(loads)*0.025,
                        f"{miss_rate:.1f}%",
                        ha="center", va="bottom", fontsize=14, fontweight="bold", 
                        color="#c0392b", rotation=90)

            ax.set_xlabel("Configuration", fontsize=11, fontweight="bold")
            ax.set_ylabel("Event count", fontsize=11, fontweight="bold")
            ax.set_title(f"L1 cache events – Input size = {size}", fontsize=13, fontweight="bold", pad=12)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=14)
            ax.legend(fontsize=15, loc="upper right", framealpha=0.95)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"benchmark_size_{size}.png"), dpi=200, bbox_inches="tight")

    # ==================== GESTIONE SPECIALE PER BASELINE (UNICO GRAFICO CACHE MISS) ====================
    # Verifichiamo se stiamo analizzando una singola versione con una sola configurazione per size
    is_single_config_per_size = False
    if len(version_groups) == 1 and args.mode == 'single':
        # Prendiamo la prima (e unica) versione
        version_name = list(version_groups.keys())[0]
        runs_in_version = version_groups[version_name]
        
        # Raggruppiamo per size
        size_groups = defaultdict(list)
        for run in runs_in_version:
            size_groups[run["size"]].append(run)
        
        # Verifichiamo se per ogni size c'è una sola configurazione
        is_single_config_per_size = all(len(configs) == 1 for configs in size_groups.values())
    
    # ==================== GRAFICO SPECIALE: CACHE MISS CON TUTTE LE CONFIGURAZIONI (SOLO SE SEMPLICE) ====================
    # Verifica se tutte le versioni hanno kernel="N/A", lmul="1", unroll="1"
    all_simple = True
    for run in all_runs:
        if run["kernel"] != "N/A" and run["kernel"] != "0" or run["lmul"] != "1" or run["unroll"] != "1":
            all_simple = False
            break

    if all_simple and not args.no_version_size and args.mode == 'compare':
        plt.figure(figsize=(12, 7))
        
        # Raggruppa per size
        size_groups = defaultdict(list)
        for run in all_runs:
            size_groups[run["size"]].append(run)
        
        # Ordiniamo per size
        sorted_sizes = sorted(size_groups.keys())
        
        # Prepariamo i dati
        sizes = []
        loads = []
        misses = []
        for size in sorted_sizes:
            # Raggruppiamo per versione
            version_data = size_groups[size]
            for run in version_data:
                sizes.append(f"{size}\n{run['version']}")
                loads.append(run["loads"])
                misses.append(run["misses"])
        
        x = np.arange(len(sizes))
        bar_width = 0.35
        
        fig, ax = plt.subplots(figsize=(max(10, len(sizes) * 0.8), 7))
        
        # Barre per loads e misses
        bars1 = ax.bar(x - bar_width/2, loads, bar_width, 
                      label="L1 loads", color="#27ae60", alpha=0.85, 
                      edgecolor='black', linewidth=0.6)
        bars2 = ax.bar(x + bar_width/2, misses, bar_width, 
                      label="L1 misses", color="#c0392b", alpha=0.85, 
                      edgecolor='black', linewidth=0.6)
        
        # Aggiungi percentuale miss rate sopra le barre misses
        for i, (load, miss) in enumerate(zip(loads, misses)):
            miss_rate = (miss / load * 100) if load else 0
            ax.text(x[i] + bar_width/2, miss + max(loads)*0.025,
                    f"{miss_rate:.1f}%",
                    ha="center", va="bottom", fontsize=10, 
                    color="#c0392b", rotation=90)
        
        ax.set_xlabel("Input size & Version", fontsize=11, fontweight="bold")
        ax.set_ylabel("Event count", fontsize=11, fontweight="bold")
        ax.set_title("L1 cache events - All versions & sizes", 
                    fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "benchmark_all_cache_events.png"), dpi=200, bbox_inches="tight")

    # ==================== GRAFICI A BARRE PER (VERSIONE X SIZE) ====================
    if not args.no_version_size:
        # Raggruppa per versione e size
        version_size_groups = defaultdict(list)
        for run in all_runs:
            version_size_groups[(run["version"], run["size"])].append(run)
        
        if is_single_config_per_size:
            # ==================== GRAFICO UNICO PER BASELINE ====================
            # Prendiamo la prima (e unica) versione
            version_name = list(version_groups.keys())[0]
            runs_in_version = version_groups[version_name]
            
            # Raggruppiamo per size
            size_groups = defaultdict(list)
            for run in runs_in_version:
                size_groups[run["size"]].append(run)
            
            # Ordiniamo per size
            sorted_sizes = sorted(size_groups.keys())
            
            # Prepariamo i dati
            sizes = []
            loads = []
            misses = []
            for size in sorted_sizes:
                run = size_groups[size][0]  # Prendiamo l'unica configurazione per questa size
                sizes.append(size)
                loads.append(run["loads"])
                misses.append(run["misses"])
            
            # Creiamo il grafico
            x = np.arange(len(sizes))
            bar_width = 0.35
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Barre per loads e misses
            bars1 = ax.bar(x - bar_width/2, loads, bar_width, 
                          label="L1 loads", color="#27ae60", alpha=0.85, 
                          edgecolor='black', linewidth=0.6)
            bars2 = ax.bar(x + bar_width/2, misses, bar_width, 
                          label="L1 misses", color="#c0392b", alpha=0.85, 
                          edgecolor='black', linewidth=0.6)
            
            # Aggiungi percentuale miss rate sopra le barre misses (ruotata)
            for i, (load, miss) in enumerate(zip(loads, misses)):
                miss_rate = (miss / load * 100) if load else 0
                ax.text(x[i] + bar_width/2, miss + max(loads)*0.025,
                        f"{miss_rate:.1f}%",
                        ha="center", va="bottom", fontsize=14, fontweight="bold", 
                        color="#c0392b", rotation=90)
            
            ax.set_xlabel("Input size", fontsize=11, fontweight="bold")
            ax.set_ylabel("Event count", fontsize=11, fontweight="bold")
            ax.set_title(f"L1 cache events - {version_name} (all sizes)", 
                        fontsize=13, fontweight="bold", pad=12)
            ax.set_xticks(x)
            ax.set_xticklabels([str(size) for size in sizes], rotation=0, fontsize=10)
            ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
            
            plt.tight_layout()
            # Sanifica il nome della versione per il filename
            safe_version = version_name.replace(" ", "_").replace("/", "_").replace(".", "_")
            plt.savefig(os.path.join(args.output_dir, f"benchmark_{safe_version}_all_sizes.png"), dpi=200, bbox_inches="tight")
            
        else:
            # ==================== GRAFICI A BARRE PER (VERSIONE X SIZE) NORMALI ====================
            # Genera un grafico per ogni combinazione (versione, size)
            for (version, size), runs_subset in sorted(version_size_groups.items()):
                # Ordinamento funziona con LMUL negativi grazie a int()
                runs_subset.sort(key=lambda r: (
                    int(r["kernel"]) if r["kernel"].isdigit() else 0,
                    int(r["lmul"]) if r["lmul"].lstrip('-').isdigit() else 0,
                    int(r["unroll"]) if r["unroll"].isdigit() else 0
                ))
                # Etichette con LMUL frazionale formattato
                labels = []
                for run in runs_subset:
                    try:
                        lmul_val = int(run['lmul'])
                        if lmul_val < 0:
                            frac_map = {-2: "1/2", -4: "1/4", -8: "1/8"}
                            lmul_display = frac_map.get(lmul_val, f"1/{abs(lmul_val)}")
                        else:
                            lmul_display = str(lmul_val)
                    except:
                        lmul_display = run['lmul']
                    
                    unroll_suffix = f" UNROLL={run['unroll']}" if run['unroll'] != "1" else ""
                    labels.append(f"k={run['kernel']} LMUL={lmul_display}{unroll_suffix}")

                loads = [run["loads"] for run in runs_subset]
                misses = [run["misses"] for run in runs_subset]

                x = np.arange(len(labels))
                bar_width = 0.35

                fig, ax = plt.subplots(figsize=(10.2, 6.5))

                # Barre per loads e misses
                bars1 = ax.bar(x - bar_width/2, loads, bar_width, 
                              label="L1 loads", color="#27ae60", alpha=0.85, 
                              edgecolor='black', linewidth=0.6)
                bars2 = ax.bar(x + bar_width/2, misses, bar_width, 
                              label="L1 misses", color="#c0392b", alpha=0.85, 
                              edgecolor='black', linewidth=0.6)

                # Aggiungi percentuale miss rate sopra le barre misses (ruotata)
                for i, (load, miss) in enumerate(zip(loads, misses)):
                    miss_rate = (miss / load * 100) if load else 0
                    ax.text(x[i] + bar_width/2, miss + max(loads)*0.025,
                            f"{miss_rate:.1f}%",
                            ha="center", va="bottom", fontsize=14, fontweight="bold", 
                            color="#c0392b", rotation=90)

                ax.set_xlabel("Configuration (kernel, LMUL, UNROLL)", fontsize=11, fontweight="bold")
                ax.set_ylabel("Event count", fontsize=11, fontweight="bold")
                ax.set_title(f"L1 cache events – {version} – Input size = {size}", 
                            fontsize=13, fontweight="bold", pad=12)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
                ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
                ax.grid(axis="y", alpha=0.3, linestyle="--")
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

                plt.tight_layout()
                # Sanifica il nome della versione per il filename
                safe_version = version.replace(" ", "_").replace("/", "_").replace(".", "_")
                plt.savefig(os.path.join(args.output_dir, f"benchmark_{safe_version}_size_{size}.png"), dpi=200, bbox_inches="tight")

    # ==================== OUTPUT FINALE ====================
    total_plots = 0
    if not args.no_global and args.mode == 'compare':
        total_plots += 1
    if not args.no_version:
        total_plots += len(version_groups)
    if not args.no_size and args.mode == 'compare':
        total_plots += len(set(run["size"] for run in all_runs))
    if not args.no_version_size:
        if is_single_config_per_size:
            total_plots += 1  # Un solo grafico speciale per la baseline
        else:
            total_plots += len(version_size_groups)
    if all_simple and args.mode == 'compare' and not args.no_version_size:
        total_plots += 1  # Grafico speciale per cache miss con tutte le configurazioni

    print(f"\n✓ Generati {total_plots} grafici:")
    if not args.no_global and args.mode == 'compare':
        print("  - 1 grafico globale (tutte le configurazioni) con legenda multi-colonna")
    if not args.no_version:
        print(f"  - {len(version_groups)} grafici tempi per singola versione")
    if not args.no_size and args.mode == 'compare':
        print(f"  - {len(set(run['size'] for run in all_runs))} grafici cache events per size (tutte le versioni)")
    if not args.no_version_size:
        if is_single_config_per_size:
            print("  - 1 grafico speciale per la baseline con tutte le size")
        else:
            print(f"  - {len(version_size_groups)} grafici cache events per (versione × size)")
    if all_simple and args.mode == 'compare' and not args.no_version_size:
        print("  - 1 grafico speciale per cache miss con tutte le configurazioni")
    print("\n✓ Supporto LMUL frazionali: -2→1/2, -4→1/4, -8→1/8")
    print("✓ Supporto UNROLL: mostrato solo se diverso da 1")
    print(f"✓ Tutti i grafici salvati nella directory: {args.output_dir}")
    print("✓ Tutti i grafici salvati come PNG ad alta risoluzione (200 DPI)")

if __name__ == "__main__":
    main()