import pandas as pd
from dbfread import DBF
import csv
import os
from tqdm import tqdm
import gc

def dbf_to_csv_memory_efficient(dbf_path, csv_path, chunk_size=10000, encoding='utf-8'):
    """
    Converte un file DBF in CSV gestendo file di grandi dimensioni in modo memory-efficient.
    
    Args:
        dbf_path (str): Percorso del file DBF di input
        csv_path (str): Percorso del file CSV di output
        chunk_size (int): Numero di record da processare per volta
        encoding (str): Encoding del file (utf-8, latin-1, cp1252, etc.)
    
    Returns:
        dict: Statistiche della conversione
    """
    
    print(f"Avvio conversione: {dbf_path} -> {csv_path}")
    
    try:
        # Apri il file DBF
        dbf = DBF(dbf_path, encoding=encoding, lowernames=True)
        
        # Ottieni informazioni sul file
        total_records = len(dbf)
        field_names = dbf.field_names
        
        print(f"Record totali: {total_records:,}")
        print(f"Campi: {len(field_names)} - {field_names[:5]}{'...' if len(field_names) > 5 else ''}")
        
        # Crea il file CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            
            # Processa in chunk per gestire file grandi
            records_processed = 0
            
            with tqdm(total=total_records, desc="Conversione") as pbar:
                chunk = []
                
                for record in dbf:
                    # Converte None in stringa vuota e gestisce tipi di dati
                    clean_record = {}
                    for key, value in record.items():
                        if value is None:
                            clean_record[key] = ''
                        elif isinstance(value, bytes):
                            # Gestisce campi binary/memo
                            clean_record[key] = value.decode(encoding, errors='ignore')
                        else:
                            clean_record[key] = str(value)
                    
                    chunk.append(clean_record)
                    
                    # Scrivi chunk quando raggiunge la dimensione target
                    if len(chunk) >= chunk_size:
                        writer.writerows(chunk)
                        records_processed += len(chunk)
                        pbar.update(len(chunk))
                        chunk = []
                        
                        # Libera memoria
                        gc.collect()
                
                # Scrivi l'ultimo chunk
                if chunk:
                    writer.writerows(chunk)
                    records_processed += len(chunk)
                    pbar.update(len(chunk))
        
        # Statistiche finali
        csv_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        dbf_size_mb = os.path.getsize(dbf_path) / (1024 * 1024)
        
        stats = {
            'records_processed': records_processed,
            'fields_count': len(field_names),
            'dbf_size_mb': round(dbf_size_mb, 2),
            'csv_size_mb': round(csv_size_mb, 2),
            'success': True
        }
        
        print(f"\n✅ Conversione completata!")
        print(f"Record processati: {records_processed:,}")
        print(f"Dimensione DBF: {dbf_size_mb:.1f} MB")
        print(f"Dimensione CSV: {csv_size_mb:.1f} MB")
        
        return stats
        
    except Exception as e:
        print(f"❌ Errore durante la conversione: {str(e)}")
        return {'success': False, 'error': str(e)}

def dbf_to_csv_pandas(dbf_path, csv_path, encoding='utf-8'):
    """
    Versione alternativa usando pandas (più veloce ma usa più memoria).
    Usa solo per file più piccoli o se hai RAM sufficiente.
    
    Args:
        dbf_path (str): Percorso del file DBF
        csv_path (str): Percorso del file CSV di output  
        encoding (str): Encoding del file
    """
    
    print(f"Conversione con pandas: {dbf_path} -> {csv_path}")
    
    try:
        # Leggi il DBF
        dbf = DBF(dbf_path, encoding=encoding, lowernames=True)
        
        # Converti in DataFrame
        df = pd.DataFrame(iter(dbf))
        
        print(f"Dimensioni DataFrame: {df.shape}")
        print(f"Memoria utilizzata: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Salva in CSV
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        csv_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"✅ File CSV creato: {csv_size_mb:.1f} MB")
        
        return {'success': True, 'records': len(df), 'csv_size_mb': csv_size_mb}
        
    except Exception as e:
        print(f"❌ Errore: {str(e)}")
        return {'success': False, 'error': str(e)}

def detect_dbf_encoding(dbf_path):
    """
    Prova a rilevare l'encoding del file DBF testando encoding comuni.
    
    Args:
        dbf_path (str): Percorso del file DBF
        
    Returns:
        str: Encoding rilevato o 'utf-8' come default
    """
    
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'cp850']
    
    for encoding in encodings_to_try:
        try:
            dbf = DBF(dbf_path, encoding=encoding)
            # Prova a leggere i primi record
            sample = list(dbf)[:10]
            print(f"✅ Encoding rilevato: {encoding}")
            return encoding
        except (UnicodeDecodeError, Exception):
            continue
    
    print("⚠️ Encoding non rilevato, uso utf-8 come default")
    return 'utf-8'

def main():
    """Esempio d'uso delle funzioni"""
    
    # Esempio di conversione
    dbf_file = "/home/aress/Scaricati/PCODE_2024_PT/PCODE_2024_PT.dbf"  # Sostituisci con il tuo file
    csv_file = "output_file.csv"
    
    # Verifica che il file esista
    if not os.path.exists(dbf_file):
        print(f"⚠️ File non trovato: {dbf_file}")
        print("Modifica la variabile 'dbf_file' con il percorso corretto")
        return
    
    # Rileva encoding
    encoding = detect_dbf_encoding(dbf_file)
    
    # Conversione memory-efficient (consigliata per file grandi)
    stats = dbf_to_csv_memory_efficient(
        dbf_path=dbf_file,
        csv_path=csv_file,
        chunk_size=5000,  # Riduci se hai poca RAM
        encoding=encoding
    )
    
    if stats['success']:
        print(f"\n📊 Statistiche conversione:")
        for key, value in stats.items():
            if key != 'success':
                print(f"  {key}: {value}")

# Installazione dipendenze (esegui nel terminale):
# pip install dbfread pandas tqdm

if __name__ == "__main__":
    main()