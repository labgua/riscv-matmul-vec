#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import argparse, sys, os
import numpy as np
from itertools import cycle

plt.style.use('seaborn-v0_8-darkgrid')

p = argparse.ArgumentParser(description="Plot speedup con griglia e tick espliciti")
p.add_argument("csv", help="File CSV con speedup")
p.add_argument("-g", "--groupby", help="Colonne che distinguono le serie (es: 'kernel,lmul')")
p.add_argument("-o", "--out", default="speedup.png", help="File output")
p.add_argument("-x", default="size", help="Colonna asse X (default: size)")
p.add_argument("-y", default="speedup", help="Colonna asse Y (default: speedup)")
p.add_argument("--exclude-param", help="Parametro da filtrare (es: 'lmul')")
p.add_argument("--exclude-value", type=float, help="Valore da escludere per quel parametro (es: 1.0)")
args = p.parse_args()

# Carica e pulisci
if not os.path.exists(args.csv):
    print(f"Errore: '{args.csv}' non trovato"); sys.exit(1)
df = pd.read_csv(args.csv)
df.columns = df.columns.str.strip()
df = df.dropna(subset=[args.x, args.y])

# === FILTRA DUPLICATI BASATI SUL PARAMETRO VARIATO ===
if args.exclude_param and args.exclude_value is not None:
    if args.exclude_param not in df.columns:
        print(f"Errore: parametro '{args.exclude_param}' non trovato nel CSV")
        sys.exit(1)
    
    initial_rows = len(df)
    
    # Per ogni combinazione di size+kernel, se esistono righe con exclude_value 
    # E anche righe con valori diversi, rimuovi quelle con exclude_value
    group_cols = [args.x] + [g.strip() for g in args.groupby.split(',')] if args.groupby else [args.x]
    
    # Trova le combinazioni che hanno sia il valore da escludere che altri valori
    df['has_other_values'] = df.groupby(group_cols[:-1] if args.exclude_param in group_cols else group_cols)[args.exclude_param] \
        .transform(lambda x: (x != args.exclude_value).any() and (x == args.exclude_value).any())
    
    # Rimuovi le righe dove has_other_values=True E il parametro ha il valore da escludere
    mask = (df['has_other_values'] == True) & (df[args.exclude_param] == args.exclude_value)
    df = df[~mask].drop(columns=['has_other_values'])
    
    removed = initial_rows - len(df)
    print(f"Filtrati {removed} duplicati con {args.exclude_param}={args.exclude_value}")

# Setup grafico
fig, ax = plt.subplots(figsize=(10, 6))

# Gruppi per le serie
groups = [g.strip() for g in args.groupby.split(',')] if args.groupby else []

# Colori, marker e tratteggi
colors = plt.cm.tab20.colors
markers = ['o', 's', '^', 'v', 'D', 'x', '+', '*', 'p', 'h', '8', '<', '>', 'P', 'X', 'd']
linestyles = ['-', '--', '-.', ':']

if not groups:
    ax.plot(df[args.x], df[args.y], marker='o', linewidth=2, label=args.y)
else:
    grouped = df.groupby(groups)
    n_series = len(grouped)
    
    color_cycle = cycle(colors)
    marker_cycle = cycle(markers)
    linestyle_cycle = cycle(linestyles)
    
    for i, (name, grp) in enumerate(grouped):
        label = ', '.join(f"{k}={v}" for k, v in zip(groups, name if isinstance(name, tuple) else [name]))
        color = next(color_cycle)
        marker = next(marker_cycle)
        linestyle = next(linestyle_cycle)
        ax.plot(grp[args.x], grp[args.y], marker=marker, linestyle=linestyle, 
                linewidth=2, label=label, color=color, markersize=6)
    
    ncol = 2 if n_series > 8 else 1
    ax.legend(fontsize=9, frameon=True, edgecolor='black', ncol=ncol, loc='best')

# Linea di riferimento Y=1
ax.axhline(y=1, color='red', linestyle='-', linewidth=2, label='Baseline (1.00)', zorder=10)

# Tick X
x_ticks = sorted(df[args.x].unique())
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks, rotation=45, ha='right', fontsize=9)

# Tick Y
y_min, y_max = df[args.y].min(), df[args.y].max()
y_ticks = list(ax.get_yticks())
if 1.0 not in y_ticks:
    y_ticks.append(1.0)
    y_ticks = sorted(y_ticks)
ax.set_yticks(y_ticks)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.grid(True, which='both', axis='both', linestyle='--', alpha=0.7)

ax.set_xlabel(args.x.capitalize(), fontsize=11, fontweight='bold')
ax.set_ylabel(args.y.capitalize(), fontsize=11, fontweight='bold')
ax.set_title(f"Speedup - {os.path.basename(args.csv)}", fontsize=13, pad=15)

plt.tight_layout()
plt.savefig(args.out, dpi=300, bbox_inches='tight')
print(f"✓ Salvato: {args.out} ({n_series if groups else 1} serie)")


'''



python3 plot_speedup.py ../BENCH-dati-finali/5_tiling_v3/confronto-autovect/table-tiling_v3-autovect-speedup.csv -o ../BENCH-dati-finali/5_tiling_v3/confronto-autovect/speedup.png -g "kernel,lmul" --exclude-param "lmul" --exclude-value 1.0


python3 plot_speedup.py ../BENCH-dati-finali/6_reordered_tiling/confronto-tiling_v3/table-reordered_tiling-tiling_v3-speedup.csv -o ../BENCH-dati-finali/6_reordered_tiling/confronto-tiling_v3/speedup.png -g "kernel,lmul" --exclude-param "lmul" --exclude-value 1.0



 python3 plot_speedup.py ../BENCH-dati-finali/7_reordered_tiling_unrolling/confronto-reordered-tiling/table-reordered_tiling_unrolling-reordered_tiling-speedup.csv -o ../BENCH-dati-finali/7_reordered_tiling_unrolling/confronto-reordered-tiling/speedup.png -g "kernel,lmul,unroll" --exclude-param "unroll" --exclude-value 1


'''