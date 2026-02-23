import pandas as pd
import argparse
import sys
import os
import re

def parse_lmul_value(val):
    """
    Converte un valore LMUL che può essere:
    - Intero: "1", "2", "4", "8"
    - Frazione: "1/2", "1/4", "1/8"
    - Float: "1.0", "2.0"
    - Già numerico: 1, 2, 0.5
    Restituisce il valore numerico float.
    """
    if pd.isna(val):
        return 1.0
    
    # Se è già numerico
    if isinstance(val, (int, float)):
        return float(val)
    
    val = str(val).strip()
    
    # Se è una frazione (es. "1/2")
    if '/' in val:
        try:
            num, den = val.split('/')
            return float(num) / float(den)
        except:
            return 1.0
    
    # Altrimenti prova a convertire come numero
    try:
        return float(val)
    except:
        return 1.0

def format_lmul_value(val):
    """
    Formatta un valore LMUL per la scrittura nel CSV:
    - Se è intero (1.0, 2.0, 4.0, 8.0) → "1", "2", "4", "8"
    - Se è frazionario (0.5, 0.25, 0.125) → "1/2", "1/4", "1/8"
    - Altrimenti → stringa normale
    """
    if pd.isna(val):
        return "1"
    
    val = float(val)
    
    # Mappatura frazioni comuni
    fraction_map = {
        0.125: "1/8",
        0.25: "1/4",
        0.5: "1/2",
        1.0: "1",
        2.0: "2",
        4.0: "4",
        8.0: "8"
    }
    
    # Cerca corrispondenza esatta o molto vicina
    for frac_val, frac_str in fraction_map.items():
        if abs(val - frac_val) < 1e-9:
            return frac_str
    
    # Se non è un valore standard, formatta come float
    if val == int(val):
        return str(int(val))
    else:
        return str(val)

def load_csv(filepath):
    """Carica il CSV e pulisce i nomi delle colonne."""
    if not os.path.exists(filepath):
        print(f"Errore: Il file {filepath} non esiste.")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df

def apply_defaults(df, defaults):
    """Applica i default alle colonne se mancanti (modifica il df in place)."""
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            # Usa parse_lmul_value per gestire frazioni
            if col in ['lmul']:
                df[col] = df[col].apply(parse_lmul_value)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(val)
    return df

def main():
    parser = argparse.ArgumentParser(description="Calcola Speedup mantenendo intatta la struttura del file B.")
    parser.add_argument("file_base", help="File CSV Baseline (A)")
    parser.add_argument("file_opt", help="File CSV Ottimizzata (B)")
    parser.add_argument("-o", "--output", default="result_speedup.csv", help="File di output")
    parser.add_argument("-m", "--metric", default="time", help="Colonna tempo/cicli (default: 'time')")
    parser.add_argument("-k", "--keys", default="size,kernel,lmul,unroll", help="Chiavi per matching (csv)")
    
    args = parser.parse_args()
    defaults = {'lmul': 1, 'unroll': 1}

    # 1. Caricamento File B (Ottimizzata)
    print(f"Caricamento B: {args.file_opt}...")
    df_opt = load_csv(args.file_opt)
    
    # 2. SALVA COLONNE ORIGINALI ORA (Prima di aggiungere default)
    original_opt_columns = df_opt.columns.tolist()
    
    # Salva anche i valori originali di lmul per mantenerli nel formato corretto
    if 'lmul' in df_opt.columns:
        original_lmul_values = df_opt['lmul'].copy()
    else:
        original_lmul_values = None

    # 3. Applica default per il calcolo interno su B
    df_opt = apply_defaults(df_opt, defaults)

    # 4. Caricamento File A (Baseline)
    print(f"Caricamento A: {args.file_base}...")
    df_base = load_csv(args.file_base)
    # Applica default anche su A per garantire matching coerente
    df_base = apply_defaults(df_base, defaults)

    # 5. Preparazione Chiavi
    join_keys = [k.strip() for k in args.keys.split(',')]
    
    # Check esistenza chiavi (ora dovrebbero esserci grazie ai default)
    for key in join_keys:
        if key not in df_base.columns or key not in df_opt.columns:
            print(f"Errore: Chiave '{key}' mancante anche dopo applicazione default.")
            sys.exit(1)
            
    if args.metric not in df_base.columns or args.metric not in df_opt.columns:
        print(f"Errore: Colonna metrica '{args.metric}' non trovata.")
        sys.exit(1)

    # 6. Merge
    cols_to_merge = join_keys + [args.metric]
    print("Calcolo speedup...")
    
    df_merged = df_opt.merge(
        df_base[cols_to_merge], 
        on=join_keys, 
        how='left', 
        suffixes=('', '_base')
    )

    # 7. Calcolo
    col_base = f"{args.metric}_base"
    df_merged['speedup'] = round(df_merged[col_base] / df_merged[args.metric], 3)
    df_merged['speedup'] = df_merged['speedup'].replace([float('inf'), -float('inf')], float('nan'))
    
    missing = df_merged[col_base].isna().sum()
    if missing > 0:
        print(f"[Attenzione] {missing} righe senza match in Baseline.")

    # 8. OUTPUT FILTRATO
    # Prendi solo le colonne originali di B + speedup
    final_columns = original_opt_columns + ['speedup']
    
    # Controllo di sicurezza
    for col in final_columns:
        if col not in df_merged.columns:
            print(f"Errore: Colonna '{col}' attesa non trovata nel risultato.")
            sys.exit(1)

    df_final = df_merged[final_columns].copy()
    
    # 9. FORMATTAZIONE LMUL PER L'OUTPUT
    # Ripristina il formato originale dei valori LMUL
    if 'lmul' in df_final.columns:
        df_final['lmul'] = df_final['lmul'].apply(format_lmul_value)
    
    # Crea la directory di output se non esiste
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df_final.to_csv(args.output, index=False)
    print(f"Salvato in: {args.output}")
    print(f"Speedup Medio: {df_final['speedup'].mean():.4f}")


if __name__ == "__main__":
    main()