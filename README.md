# Istruzioni da aggiornare vecchio README
Dataminer: Tool per il mining di dataset
- test_nord.py: esempio di main, modificare le impostazioni come indicato. E' possibile replicare lo stesso main su più file .py per esecuzione parallela.
- merge.py: Tool di merge, modificare la cartella di input inserendo la stessa cartella delle esecuzioni all'interno del main
- utility.py : Contiene tutte le funzioni di ritaglio, creazione mappa e creazione CSV, nonché il mapping tra le classi di GTRSB e i nomi delle label impostate da Mapillary

predictionToolV2: Addestramento modello e classificatore immagini. E' presente anche il dataset GTRSB in formato pickle (traffic-signs-data) e i nomi delle classi presenti (signnames.csv)
- Classifier.py: contiene il codice per l'addestramento del modello e i metodi per classificare le immagini.
NB!!! - E' importante che il dataset su cui si voglia fare analisi sia completo, deve esserci il file annotations.csv all'interno del dataset, inserire il percorso del file all'interno del corrente file.
In Saved_Models c'è il modello preaddestrato che utilizza il tool (VGGNet a 12 livelli).

analizerTool: Tool che effettua i vari calcoli e crea i grafici partendo dal csv generato dal tool precedente (quello di output quindi, dopo che è stato effettuato il merge).

Test2: Dataset di segnali stradali italiani

ALTRI FILE
- custom_config.txt : file che contiene la configurazione custom dei segnali stradali. Utilizzando questo file di configurazione (o modificandolo) scarica solo i segnali le cui label sono contenute in questo file. Nella mia configurazione ci sono tutti i segnali appartenenti alle 43 classi del GTRSB

## Istruzioni aggiuntive nuova disposizione
Per utilizzare gli script Python occorre creare un file `.env` a livello di progetto, inserendo i valori indicati dagli esempi del file `.env.dummy`.

### Configurazione librerie
Si consiglia di gestire le librerie in ambienti virtuali, come offerto da python venv.
```bash
# Creazione ambiente (da eseguire nella root del progetto)
python -m venv .venv

# Attivazione (Windows)
.venv\Scripts\activate

# Attivazione (Linux/macOS)
source .venv/bin/activate

# Installazione dipendenze
pip install -r requirements.txt

# Disattivazione
deactivate
```
Si consiglia caldamente di non committare la cartella associata all'ambiente virtuale.