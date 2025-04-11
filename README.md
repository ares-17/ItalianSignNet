# Datamining e classificazione cartelli stradali italiani

## Intro
Il progetto costituisce un esempio di raccolta dei cartelli stradali per un territorio ben delineato e con confini complessi e strutturati come l'Italia. 

Gli script raccolti sono divisi per: raccolta aree geografiche di interesse, utilizzo delle API di Mapillary e raccolta dei cartelli stradali a partire dalle immagini scaricate.
Le bounding boxes considerate al momento sono:

![Bounding boxes Italiane](public/bounding_boxes_italia_11_02_25.png "Bounding boxes Italiane")

## Caratteristiche del Progetto

- **Multithreading ad Alte Prestazioni**  
  Il cuore del sistema è l'elaborazione parallela: le bounding boxes vengono processate simultaneamente grazie al multithreading. Questo approccio riduce drasticamente i tempi di attesa delle chiamate API e massimizza l'utilizzo della banda disponibile, consentendo di raccogliere dati su larga scala anche su hardware modesto. I thread sono utilizzati sia sull'esecuzione parallela di più bounding boxes che sull'esecuzione delle numerose API Mapillary coinvolte per ciascuna bounding box

- **Generazione Dinamica delle Bounding Boxes**  
  Partendo da una macro-area definita (`INITIAL_BBOX`), il sistema suddivide il territorio in celle, escludendo quelle che ricadono fuori dall'Italia grazie all'uso di file geojson dei confini nazionali.

- **Dataset Strutturato e Modulabile**  
  Il progetto organizza i dati in cartelle specifiche:  
  - `geojson_folder`: File GeoJSON che definiscono la griglia di raccolta  
  - `annotations_image`: File JSON contenenti annotazioni dettagliate per ogni immagine  
  - `resized_images`: Immagini ritagliate dei cartelli stradali  
  - `signal_catalog`: CSV con informazioni approfondite sui cartelli  
  - `images`: Immagini originali scaricate  
  - `annotations.csv`: Mappa tra filename ed etichette

## Installazione
Scaricare il progetto e configurare le variabili di ambiente nel file `.env` da creare a partire dalla copia di esempio `.env.dummy`. Nel file sono presenti valori di defaults che per un uso basico del progetto possono essere sufficienti. 

Nota: al momento i file in `src/model/traffic-signs-data` che costituiscono il dataset del modello non sono caricati su GitHub per eccedenza dei limiti di spazio.

### Variabili d'ambiente
Le variabili di ambiente sono utilizzate dai principali script Python e aiutano alla generalizzazione e al basso accoppiamento tra diversi file.

Le variabili di ambiente sono lette da `.env` che deve esser generato a partire da `.env.dummy`. Di seguito una breve introduzione delle variabili:

| Variabile     | Descrizione      |
| ------------- | ------------- |
| BASE_DIR | Cartella root del progetto  |
| MAPILLARY_API_KEY | Personal Token di Mapillary  |
| GEOJSON_ITALY_PATH | File geojson dei confini. Attualmente l'unico presente è `src/resources/limits_IT_regions.geojson` |
| DOWNLOAD_BBOX_IMAGES_PATH | Percorso assoluto che indirizza al file `src/dataminer/download_bbox_images.py` |
| MERGE_FULLSIZE_IMAGES | Booleano utilizzato da `src/utils/merge.py` che se vero implica l'unione delle cartelle dei bounding boxes contenenti le immagini scaricate da Mapillary. Per risparmiare spazio e tempo, lasciare il valore a `False`. Se si intende valorizzarlo con valore positivo, occorre modificare lo script `src/dataminer/download_bbox_images.py` che di default elimina i file non utili alla creazione del dataset |
| MERGE_BOUNDED_IMAGES | Allo stesso modo della variabile precendete, valorizzarlo solo nei casi in cui siano necessari ulteriori dati, impattando sullo spazio occupato del sistema |
| INITIAL_BBOX | Coordinate iniziali che delimitano la macro-regione nella quale sono derivate le bounding boxes |
| BBOXES_TO_REMOVE | Lista di ID delle bounding boxes da escludere. Per il territorio italiano sono escluse tutte quelle che ricadono interamente nel Mar Mediterraneo oppure interamente su territori esteri |
| LAT_STEP_BBOX | Dimensioni delle bounding boxes |
| LON_STEP_BBOX | Dimensioni delle bounding boxes |
| NUM_FEATURES_BBOX | Numero massimo di feature considerate per ogni immagine scaricata da Mapillary |
| MAX_PARALLEL_EXECUTIONS | Numero di bounding boxes eseguite contemporaneamente. Il valore di default è "4" che non costituisce un parametro valido ed efficiente per ogni architettura hardware |
| DATAMINER_GRID_WORKERS | Numero di bounding boxes eseguite contemporaneamente. Il valore di default è "8" che sembra essere il massimo numero di chiamate eseguibile con Mpaillary e con la stessa sessione HTTP  |


### Librerie
Nel progetto le dipendenze sono raccolte in `requirements.txt`. Si consiglia di utilizzare un ambiente virtuale Python per gestirle. 
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

## Struttura
Nel progetto sono presenti sia script e file per la gestione e download dei cartelli stradali italiani e sia script Python per l'inferenza di un modello VGGNet su tali cartelli. La struttura parametrica del progetto consente anche di personalizzare le aree geografiche e i cartelli da considerare.
Composizione:

- src/dataminer:
   
   - **write_bboxes.py**: genera le coordinate delle regioni (bounfing boxes) con le quali suddividere il poligono originale, definito da `INITIAL_BBOX` in `.env`
   - **bounding_boxes.py**: file autogenerato da `write_bboxes.py` delle regioni geografiche,
   - **utility.py**: raccolti di funzioni e metodi che semplificano il processo di datamining dei cartelli stradali,
   - **main.py**: esegue la raccolta dei cartelli stradali avvalendosi di `download_bbox_images.py` e `Dataminer.py`; il parametro `MAX_PARALLEL_EXECUTIONS` indica il numero di regioni esegue parallelamente su tali script,
   - **download_bbox_images.py**: richiamato da `main.py` con i parametri di una regione geografica, utilizza il `Dataminer.py` per ricavare i dati e scrivere sul disco i file relativi alle immagini raccolte,
   - **Dataminer.py**: scarica le immagini dei cartelli. I cartelli considerati sono censiti in `utils.py`. Ogni regione geografica di input è suddivisa in una griglia che assicura una raccolta equa e distribuita su ogni cella della regione geografica. Per ogni cartello stradale, si verifica se questo ricade nei confini delimitati dal file `limits_IT_regions.geojson` che contiene i confini delle regioni italiane. La classe Dataminer divede la bounding box in una griglia con celle di ugual dimensione per forzare la raccolta da ciascuna aree geografica. Le API di Mapillary sono richiamate da thread definiti dal parametro `DATAMINER_GRID_WORKERS` che velocizza la raccolta dei dati

-  src/resources:

   - **limits_IT_regions.geojson**: raccolta di poligoni geografici che delimitano i confini delle regioni italiane. File scaricato l'11/02/2025 dal repository di (openpolis)[https://github.com/openpolis/geojson-italy],
   - **traffic-signs.txt**: configurazione utilizzata da `src/dataminer/Dataminer.py`

- src/utils:

   - **merge.py**: script che unisce i file generati sul disco per ciascuna bounding boxes di `src/dataminer/bounding_boxes.py`. Utile per derivare statistiche e alla base della creazione di un dataset sull'intero territorio indicato. Utile anche a visualizzare i markers dei cartelli stradali, visualizzando il file HTML generato,
   - **show_bbox_map.py**: genera la mappa HTML delle bounding boxes considerate. Il file è memorizzato in `public/bounding_boxes_italia.html`

- src/stats:

   - **analizer2.py**: script che genera grafici statistici sul modello addestrato in `src/model` sul dataset creato dai cartelli stradali considerati,
   - **count_regions_images.py**: conta i cartelli stradali suddividendoli in tre etichette che corrispondo a tre macroaree della penisola italiana

- src/model:
   - **Classifier.py**: modello VGGNet
   - **results**: risultati scritti durante l'esecuzione del modello
   - **saved**: eventuali parametri salvati del modello
