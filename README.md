# Datamining e classificazione cartelli stradali italiani
## Intro
Il progetto costituisce un esempio di raccolta dei cartelli stradali per un territorio ben delineato e con confini complessi e strutturati come l'Italia.
Gli script raccolti sono divisi per: raccolta aree geografiche di interesse, utilizzo delle API di Mapillary e raccolta dei cartelli stradali a partire dalle immagini scaricate. Le bounding boxes considerate al momento sono:

![Bounding boxes Italiane](public/bounding_boxes_italia_11_02_25.png "Bounding boxes Italiane")
## Caratteristiche del Progetto
*   **Multithreading ad Alte Prestazioni**
    Il cuore del sistema è l'elaborazione parallela: le bounding boxes vengono processate simultaneamente grazie al multithreading. Questo approccio riduce drasticamente i tempi di attesa delle chiamate API e massimizza l'utilizzo della banda disponibile, consentendo di raccogliere dati su larga scala anche su hardware modesto. I thread sono utilizzati sia sull'esecuzione parallela di più bounding boxes che sull'esecuzione delle numerose API Mapillary coinvolte per ciascuna bounding box.
*   **Generazione Dinamica delle Bounding Boxes**
    Partendo da una macro-area definita (INITIAL_BBOX), il sistema suddivide il territorio in celle, escludendo quelle che ricadono fuori dall'Italia grazie all'uso di file geojson dei confini nazionali.
*   **Dataset Strutturato e Modulabile**
    Il progetto organizza i dati in cartelle specifiche:
    *  geojson_folder: File GeoJSON che definiscono la griglia di raccolta.
    *  annotations_image: File JSON contenenti annotazioni dettagliate per ogni immagine.
    *  resized_images: Immagini ritagliate dei cartelli stradali.
    *  signal_catalog: CSV con informazioni approfondite sui cartelli.
    *  images: Immagini originali scaricate.
    *  annotations.csv: Mappa tra filename ed etichette.

## Installazione
Scaricare il progetto e configurare le variabili di ambiente nel file **.env** da creare a partire dalla copia di esempio **.env.dummy**. Nel file sono presenti valori di defaults che per un uso basico del progetto possono essere sufficienti.
Nota: al momento i file in `src/model/traffic-signs-data` che costituiscono il dataset del modello non sono caricati su GitHub per eccedenza dei limiti di spazio.

## Variabili d'ambiente
Le variabili di ambiente sono utilizzate dai principali script Python e aiutano alla generalizzazione e al basso accoppiamento tra diversi file.
Sono lette da .env e includono:
| Variabile | Descrizione |
| ------ | ------ |
| **BASE_DIR** | Cartella root del progetto. |
| **MAPILLARY_API_KEY** | Personal Token di Mapillary. |
| **GEOJSON_ITALY_PATH** | File geojson dei confini. Attualmente l'unico presente è `src/resources/limits_IT_regions.geojson`. |
| **DOWNLOAD_BBOX_IMAGES_PATH** | Percorso assoluto che indirizza al file `src/dataminer/dowload_bbox_images.py`. |
| **MERGE_FULLSIZE_IMAGES** | Booleano utilizzato da `src/dataminer/merge_bboxes_folders.py` che, se vero, implica l'unione delle cartelle dei bounding boxes contenenti le immagini scaricate da Mapillary. Per risparmiare spazio e tempo, è consigliabile lasciare il valore a `False`. Se si intende valorizzarlo con valore positivo, occorre modificare lo script `src/dataminer/dowload_bbox_images.py` che di default elimina i file non utili alla creazione del dataset. |
| **MERGE_BOUNDED_IMAGES** | Allo stesso modo della variabile precedente, valorizzarlo solo nei casi in cui siano necessari ulteriori dati, impattando sullo spazio occupato del sistema. |
| **INITIAL_BBOX** | Coordinate iniziali che delimitano la macro-regione nella quale sono derivate le bounding boxes. |
| **BBOXES_TO_REMOVE** | Lista di ID delle bounding boxes da escludere. Per il territorio italiano sono escluse tutte quelle che ricadono interamente nel Mar Mediterraneo oppure interamente su territori esteri. |
| **LAT_STEP_BBOX** | Dimensioni delle bounding boxes. |
| **LON_STEP_BBOX** | Dimensioni delle bounding boxes. |
| **NUM_FEATURES_BBOX** | Numero massimo di feature considerate per ogni immagine scaricata da Mapillary. |
| **MAX_PARALLEL_EXECUTIONS** | Numero di bounding boxes eseguite contemporaneamente. Il valore di default è "4", che non costituisce un parametro valido ed efficiente per ogni architettura hardware. |
| **DATAMINER_GRID_WORKERS** | Numero di bounding boxes eseguite contemporaneamente. Il valore di default è "8", che sembra essere il massimo numero di chiamate eseguibile con Mapillary e con la stessa sessione HTTP. |
| **DBSCAN_DISTANCE** | Distanza massima (in metri) utilizzata dall'algoritmo DBSCAN per il raggruppamento spaziale dei cartelli stradali. |
| **DBSCAN_MIN_SAMPLES** | Numero minimo di campioni (punti) richiesti per formare un cluster nell'algoritmo DBSCAN. |
| **ITALY_HEATMAP_BBOX** | Coordinate della bounding box per la visualizzazione della heatmap dei cluster sulla mappa dell'Italia nei report. |
| **TEST_CASE_BASE_ROOT** | Percorso radice per la gestione dei casi di test e l'accesso ai dati di input. |
| **SEED_GROUP_SHUFFLE_DATASET** | Seed per la riproducibilità dello shuffle dei dati durante la suddivisione del dataset in train/validation/test. |
| **TEST_SIZE_DATASET** | Percentuale dei dati da assegnare al set di test durante la suddivisione del dataset. |
| **VAL_SIZE_DATASET** | Percentuale dei dati da assegnare al set di validazione durante la suddivisione del dataset. |
| **APPLY_AUGMENTATIONS** | Variabile booleana che abilita o disabilita l'applicazione delle tecniche di data augmentation al dataset. |
| **MLFLOW_ENDPOINT** | Endpoint del server MLflow per il logging e il tracciamento degli esperimenti e dei dataset generati. |

## Librerie
Nel progetto le dipendenze sono raccolte in `requirements.txt`. Si consiglia di utilizzare un ambiente virtuale Python per gestirle.

## Struttura
Nel progetto sono presenti script e file per la gestione e il download dei cartelli stradali italiani, per la creazione e l'analisi del dataset. La struttura parametrica del progetto consente anche di personalizzare le aree geografiche e i cartelli da considerare.

*   **src/dataminer**: Contiene gli script principali per la raccolta e pre-elaborazione dei dati geospaziali da Mapillary.
    *   **write_bboxes.py**: Genera le coordinate delle regioni (bounding boxes) con le quali suddividere il poligono originale, definito dalla variabile d'ambiente `INITIAL_BBOX`.
    *   **bounding_boxes.py**: File auto-generato da `write_bboxes.py` che contiene le definizioni delle regioni geografiche sotto forma di un dizionario `BOUNDING_BOXES`.
    *   **utility.py**: Contiene una raccolta di funzioni e metodi di supporto che semplificano il processo di datamining dei cartelli stradali, inclusa la gestione delle cartelle, il calcolo delle coordinate dei bounding box, la creazione di file CSV e la generazione di mappe HTML.
    *   **main.py**: Il punto di ingresso principale per avviare l'elaborazione parallela delle regioni. Esegue la raccolta dei cartelli stradali avvalendosi di `dowload_bbox_images.py` (o `download_bbox_images.py` come nominato in altre fonti) per ciascuna bounding box, sfruttando il multithreading e il parametro `MAX_PARALLEL_EXECUTIONS`.
    *   **dowload_bbox_images.py**: Richiamato da `main.py` con i parametri di una specifica regione geografica, questo script utilizza `Dataminer.py` per ricavare i dati e scrivere sul disco i file relativi alle immagini raccolte. Gestisce la configurazione dei parametri di estrazione, la gestione delle cartelle di output, il download dei dati grezzi in formato GeoJSON, la selezione delle feature e il download di immagini/annotazioni, e il post-processing.
    *   **Dataminer.py**: Classe principale responsabile del download delle immagini dei cartelli stradali dall'API Mapillary. Permette la configurazione di parametri di download, la gestione di dati geospaziali e l'elaborazione di immagini e annotazioni. La classe divide la bounding box in una griglia di celle per una raccolta distribuita, e le chiamate API di Mapillary sono parallelizzate tramite thread definiti da `DATAMINER_GRID_WORKERS`.
    *   **merge_bboxes_folders.py**: Script che unisce il contenuto delle cartelle di output generate per ciascuna bounding box processata. Questo include la copia di immagini (a dimensione originale o con bounding box), GeoJSON, annotazioni e cataloghi di segnali, e la fusione dei file `annotations.csv` per creare un dataset consolidato.

*   **src/dataset**: Contiene gli script per la creazione, l'elaborazione, il clustering spaziale e l'analisi del dataset finale.
    *   **main.py**: Il punto di ingresso principale per la pipeline di creazione e analisi del dataset. Orchestrates l'esecuzione di `spatial_clustering`, `create_dataset`, `augmentations`, e `dbscan_reports`, verificando anche che il server MLflow sia attivo per il logging.
    *   **augmentations.py**: Implementa diverse tecniche di aumento dei dati (come alterazioni fotometriche, effetti atmosferici e occlusione) per arricchire il dataset esistente. Le immagini aumentate vengono salvate e i metadati aggiornati.
    *   **create_dataset.py**: Script principale per la preparazione del dataset finale. Le sue funzionalità includono l'assegnazione di ID di cluster alle immagini, l'aggiunta di codici ISTAT dei comuni e quartili di reddito basati sulla posizione, l'assegnazione di macro-aree geografiche (Nord/Centro/Sud), la suddivisione del dataset in set di training, validazione e test mantenendo la stratificazione dei cluster, la copia delle immagini nelle rispettive directory e il logging dettagliato delle informazioni del dataset su MLflow.
    *   **dbscan_reports.py**: Genera report grafici e statistiche sul dataset risultante dal clustering DBSCAN. Include heatmap dei centri dei cluster, grafici sulla cardinalità dei cluster, numero di cluster per etichetta, distribuzione delle feature per quartile di reddito e conteggio dei quartili per macro-area geografica.
    *   **spatial_clustering.py**: Questo script esegue il clustering dei cartelli stradali utilizzando l'algoritmo DBSCAN con metrica "haversine". Raggruppa le immagini che mostrano lo stesso cartello stradale in base alla loro vicinanza geografica e alla loro etichetta. Genera un report riassuntivo sui cluster e salva tutte le informazioni di clustering in un file JSON.

*   **src/resources**: Contiene file di configurazione e dati geografici di riferimento utilizzati da vari script del progetto.
    *   **limits_IT_regions.geojson**: Un file GeoJSON che contiene la definizione dei poligoni geografici che delimitano i confini delle regioni italiane. È stato scaricato da un repository Openpolis. Viene utilizzato per verificare se un punto geografico ricade all'interno dell'Italia e per la geocodifica regionale.
    *   **limits_IT_municipalities.geojson**: File GeoJSON che definisce i confini geografici dei comuni italiani. Utilizzato principalmente dal `MunicipalGeocoder` per associare le coordinate ai codici ISTAT dei comuni.
    *   **traffic-signs.txt**: File di configurazione che elenca i segnali stradali specifici da considerare e raccogliere, utilizzato da `src/dataminer/Dataminer.py` quando la configurazione `Type.CUSTOM` è selezionata.
    *   **Redditi_e_principali_variabili_IRPEF_su_base_comunale_CSV_2023.csv**: File CSV contenente dati sui redditi IRPEF a livello comunale, utilizzato per arricchire il dataset con informazioni socio-economiche associate alle aree geografiche.
    *   **signnames.csv**: File utilizzato per mappare i nomi dei segnali stradali a un indice numerico di classe, fondamentale per la preparazione del dataset per modelli di machine learning.

*   **src/utils**: Contiene funzioni di utilità e classi ausiliarie utilizzate in diverse parti del progetto.
    *   **write_bbox_map.py**: Script responsabile della generazione di una mappa HTML interattiva che visualizza le bounding boxes considerate nel progetto. Il file HTML generato (`bounding_boxes_italia.html`) viene salvato nella cartella `public/`.
    *   **MunicipalGeocoder.py**: Una classe per la geocodifica di coordinate geografiche, con lo scopo di ottenere il codice ISTAT del comune in cui una data coordinata ricade. Utilizza un approccio gerarchico, cercando prima la regione e poi il comune al suo interno.
    *   **RegionGeocoder.py**: Una classe che consente di geocodificare coordinate geografiche per ottenere informazioni sulla regione, inclusi il nome, la macro-area geografica (Nord, Centro, Sud, basata su una ripartizione ufficiale) e il codice ISTAT della regione stessa.

*   **src/models**: Contiene un notebook Python per l'addestramento e la validazione dei modelli Deep Learning, utilizzando i dataset prodotti precedentemente. Nell cartella `models` è presente il file `README.models.md` che descrive come avviare i docker container per utilizzare notebook Python