import csv
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib import rcParams

# Imposta font leggibili
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 9

#==================== PARSING DEL CSV ====================
filename = sys.argv[1]
runs = []

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
        
        # UNROLL: ignoriamo per ora poiché non era gestito negli script originali
        # ma possiamo aggiungerlo se necessario in futuro
        
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
        
        runs.append({
            "version": version,
            "size": size,
            "kernel": kernel,
            "lmul": lmul,
            "time": time,
            "loads": loads,
            "misses": misses,
            "miss_rate": miss_rate
        })

if not runs:
    print("Nessun dato trovato nel file CSV!")
    sys.exit(1)

#==================== FUNZIONE PER ETICHETTA VERSIONE COMPATTA ====================
def get_short_label(version, kernel, lmul):
    """
    Genera etichetta compatta con supporto LMUL frazionali:
    - LMUL positivo: "LMUL=2"
    - LMUL negativo: "-2" → "LMUL=1/2", "-4" → "LMUL=1/4", "-8" → "LMUL=1/8"
    """
    if "reordered" in version.lower():
        base = "reordered tiling"
    else:
        base = "tiling"
    
    # Converti LMUL frazionale in rappresentazione leggibile
    if lmul != "N/A":
        try:
            lmul_val = int(lmul)
            if lmul_val < 0:
                frac_map = {-2: "1/2", -4: "1/4", -8: "1/8"}
                lmul_display = frac_map.get(lmul_val, f"1/{abs(lmul_val)}")
            else:
                lmul_display = str(lmul_val)
        except:
            lmul_display = lmul
    else:
        lmul_display = "N/A"

    return f"{base} k={kernel} LMUL={lmul_display}"

#==================== DEFINIZIONE PALETTE ESTESA ====================
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

#==================== GRAFICO GLOBALE CON LEGENDA ====================
series_all = defaultdict(list)
for run in runs:
    key = (run["version"], run["kernel"], run["lmul"])
    series_all[key].append((run["size"], run["time"]))
sorted_keys = sorted(series_all.keys())
total_series = len(sorted_keys)
plt.figure(figsize=(15, 9))
for idx, key in enumerate(sorted_keys):
    version, kernel, lmul = key
    points = sorted(series_all[key], key=lambda x: x[0])
    sizes, times = zip(*points)
    # Assegna stile unico combinando colore, linea e marcatore
    color = color_cycle[idx % len(color_cycle)]
    linestyle = linestyle_cycle[idx % len(linestyle_cycle)]
    marker = marker_cycle[idx % len(marker_cycle)]

    label = get_short_label(version, kernel, lmul)
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
plt.savefig("benchmark_all_versions.png", dpi=200, bbox_inches="tight")

#==================== GRAFICI PER VERSIONE (fissata) ====================
version_groups = defaultdict(list)
for run in runs:
    version_groups[run["version"]].append(run)
for version_name, runs_in_version in sorted(version_groups.items()):
    plt.figure(figsize=(12, 7))
    config_series = defaultdict(list)
    for run in runs_in_version:
        key = (run["kernel"], run["lmul"])
        config_series[key].append((run["size"], run["time"]))

    sorted_configs = sorted(config_series.keys())
    total_local = len(sorted_configs)

    for idx, config in enumerate(sorted_configs):
        kernel, lmul = config
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
        
        label = f"k={kernel} LMUL={lmul_display}"
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
    plt.title(f"Execution time vs Input size\n({get_short_label(version_name, '*', '*').split(' k=')[0]})", 
              fontsize=14, fontweight="bold", pad=12)
    plt.grid(True, alpha=0.35, linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f"benchmark_{version_name}.png", dpi=200, bbox_inches="tight")

#==================== GRAFICI A BARRE PER OGNI SIZE (LABELS ACCORCIATE) ====================
unique_sizes = sorted(set(run["size"] for run in runs))
for size in unique_sizes:
    runs_at_size = [run for run in runs if run["size"] == size]
    if not runs_at_size:
        continue
    # Ordinamento funziona correttamente anche con LMUL negativi (int() gestisce i negativi)
    runs_at_size.sort(key=lambda r: (
        r["version"],
        int(r["kernel"]) if r["kernel"].isdigit() else 0,
        int(r["lmul"]) if r["lmul"].lstrip('-').isdigit() else 0  # gestisce anche "-2"
    ))

    # Etichette compatte per le barre (usa get_short_label che già gestisce frazionali)
    labels = [
        get_short_label(run['version'], run['kernel'], run['lmul'])
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
    plt.savefig(f"benchmark_size_{size}.png", dpi=200, bbox_inches="tight")

#==================== NUOVA SEZIONE: GRAFICI A BARRE PER (VERSIONE X SIZE) ====================
# Raggruppa per versione e size
version_size_groups = defaultdict(list)
for run in runs:
    version_size_groups[(run["version"], run["size"])].append(run)
# Genera un grafico per ogni combinazione (versione, size)
for (version, size), runs_subset in sorted(version_size_groups.items()):
    # Ordinamento funziona con LMUL negativi grazie a int()
    runs_subset.sort(key=lambda r: (
        int(r["kernel"]) if r["kernel"].isdigit() else 0,
        int(r["lmul"]) if r["lmul"].lstrip('-').isdigit() else 0
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
        labels.append(f"k={run['kernel']} LMUL={lmul_display}")

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

    ax.set_xlabel("Configuration (kernel, LMUL)", fontsize=11, fontweight="bold")
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
    plt.savefig(f"benchmark_{safe_version}_size_{size}.png", dpi=200, bbox_inches="tight")

#==================== OUTPUT FINALE ====================
total_plots = (1 +
               len(version_groups) +
               len(unique_sizes) +
               len(version_size_groups))
print(f"✓ Generati {total_plots} grafici:")
print(f"  - 1 grafico globale (tutte le configurazioni) con legenda multi-colonna")
print(f"  - {len(version_groups)} grafici tempi per singola versione")
print(f"  - {len(unique_sizes)} grafici cache events per size (tutte le versioni)")
print(f"  - {len(version_size_groups)} grafici cache events per (versione X size)")
print("\n✓ Supporto LMUL frazionali: -2→1/2, -4→1/4, -8→1/8")
print("✓ Tutti i grafici salvati come PNG ad alta risoluzione (200 DPI)")
plt.show()