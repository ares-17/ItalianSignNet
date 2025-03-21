# ItalianSignNet Cluster Reports

Clusters' distribution             |  Clusters' Heatmap          | Clusters' by label
:-------------------------:|:-------------------------:|:-------------------------:
<img src="../../public\bar_chart_cluster_distribution_example.png" alt="Clusters' distribution" width="300"> |  <img src="../../public\heatmap_clusters_example.png" alt="Clusters' Heatmap" width="280"> | <img src="../../public\bar_chart_clusters_per_label_example.png" alt="Clusters' bar chart by label" width="350">

Gli script contenuti in questa cartella sono utili alla manipolazione dei dati ottenuti mediante i file in `src/dataminer`. 

Per la configurazione, creare un file `.env` in questa cartella replicando il file di esempio `.evn.dummy` per specificare la cartella di test con `TEST_CASE_BASE_ROOT` creata nella fase precedente.

Descrizione testuale degli script:
- `spatial_clustering.py`:
    1. Sono caricate le coordinate dei cartelli dai file GeoJSON presenti nella cartella `geojson_folder`. 
    Per ciascuna feature, viene estratto l'ID (dal campo "properties.id") e le coordinate geografiche (in formato [lon, lat])

    2. Si legge il file `annotations.csv`, che associa per ogni immagine l'etichetta del cartello

    3. I record risultanti (composti da ID immagine, etichetta e coordinate convertite in radianti) vengono raggruppati per etichetta

    4. Per ciascun gruppo (stessa etichetta) viene applicato DBSCAN con metrica "haversine" e un raggio massimo di 
    `DBSCAN_DISTANCE` (default 100 metri, convertiti in radianti). In tal modo vengono creati cluster di immagini che mostrano lo stesso cartello

    5. Sono memorizzate le seguenti informazioni in un file `.json` nella cartella `logs`: le caratteristiche dei cluster e delle informazioni di reportistica

- `dbscan_reports.py`:
   Questo script legge il file JSON generato dallo script di clustering e produce grafici nella cartella `reports` per visualizzare i risultati:
   - **Heatmap geografica**: viene creata una heatmap (utilizzando un grafico a esagoni) che mostra i centri dei cluster sulla mappa dell'Italia. La heatmap è limitata da una bounding box, definita tramite l'ambiente (variabile `ITALY_HEATMAP_BBOX`).
   - **Grafico a barre per la distribuzione totale dei cluster**: mostra la distribuzione dei cluster in base alla loro cardinalità.
   - **Grafico a barre per il numero di cluster per etichetta**: evidenzia il numero di cluster rilevati per ciascuna etichetta.