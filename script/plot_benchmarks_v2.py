import re
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib import rcParams

# Imposta font leggibili
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 9

# ==================== PARSING DEI DATI ====================
filename = sys.argv[1]

record_re = re.compile(
    r"BENCHMARK_RECORD\s*:\s*([A-Za-z0-9_]+),\s*([0-9.]+),\s*([0-9]+)"
    r"(?:,\s*([0-9]+))?"   # kernel (opzionale)
    r"(?:,\s*([0-9]+))?"   # lmul (opzionale)
)

loads_re = re.compile(r"([\d,]+)\s+L1-dcache-loads")
misses_re = re.compile(r"([\d,]+)\s+L1-dcache-load-misses")

runs = []
with open(filename, "r") as f:
    content = f.read()

for rec in record_re.finditer(content):
    version = rec.group(1)
    exec_time = float(rec.group(2))
    size = int(rec.group(3))
    kernel = rec.group(4) if rec.group(4) else "N/A"
    lmul = rec.group(5) if rec.group(5) else "N/A"
    
    tail = content[rec.end():]
    loads_match = loads_re.search(tail)
    misses_match = misses_re.search(tail)
    if not loads_match or not misses_match:
        continue
    
    loads = int(loads_match.group(1).replace(",", ""))
    misses = int(misses_match.group(1).replace(",", ""))
    miss_rate = (misses / loads) * 100 if loads else 0.0
    
    runs.append({
        "version": version,
        "size": size,
        "kernel": kernel,
        "lmul": lmul,
        "time": exec_time,
        "loads": loads,
        "misses": misses,
        "miss_rate": miss_rate
    })

if not runs:
    print("Nessun dato trovato nel file di benchmark!")
    sys.exit(1)

# ==================== FUNZIONE PER ETICHETTA VERSIONE COMPATTA ====================
def get_short_label(version, kernel, lmul):
    """
    Genera etichetta compatta:
      - "tiling k=X LMUL=Y" oppure
      - "reordered tiling k=X LMUL=Y"
    """
    if "reordered" in version.lower():
        base = "reordered tiling"
    else:
        base = "tiling"
    return f"{base} k={kernel} LMUL={lmul}"

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

# ==================== GRAFICO GLOBALE CON LEGENDA ====================
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

# ==================== GRAFICI PER VERSIONE (fissata) ====================
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
        
        label = f"k={kernel} LMUL={lmul}"
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

# ==================== GRAFICI A BARRE PER OGNI SIZE (LABELS ACCORCIATE) ====================
unique_sizes = sorted(set(run["size"] for run in runs))

for size in unique_sizes:
    runs_at_size = [run for run in runs if run["size"] == size]
    if not runs_at_size:
        continue
    
    runs_at_size.sort(key=lambda r: (
        r["version"],
        int(r["kernel"]) if r["kernel"].isdigit() else 0,
        int(r["lmul"]) if r["lmul"].isdigit() else 0
    ))
    
    # Etichette compatte per le barre
    labels = [
        get_short_label(run['version'], run['kernel'], run['lmul'])
        for run in runs_at_size
    ]
    
    loads = [run["loads"] for run in runs_at_size]
    misses = [run["misses"] for run in runs_at_size]
    
    x = np.arange(len(labels))
    bar_width = 0.35
    
    #fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.3), 9.5))
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
               ha="center", va="bottom", fontsize=14, fontweight="bold", color="#c0392b", rotation=90)
    
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

# ==================== OUTPUT FINALE ====================
total_plots = 1 + len(version_groups) + len(unique_sizes)
print(f"✓ Generati {total_plots} grafici:")
print(f"  - 1 grafico globale (tutte le configurazioni) con legenda multi-colonna")
print(f"  - {len(version_groups)} grafici per singola versione")
print(f"  - {len(unique_sizes)} grafici a barre per dimensione fissa")
print("\n✓ Etichette compatte: 'tiling k=X LMUL=Y' o 'reordered tiling k=X LMUL=Y'")
print("✓ Combinazioni uniche di colore + stile linea + marcatore per ogni serie")
print("✓ Tutti i grafici salvati come PNG ad alta risoluzione (200 DPI)")
plt.show()