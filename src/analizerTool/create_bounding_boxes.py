import folium
from folium.features import DivIcon

# Configurazione
initial_bbox = (35.288961, 6.614778, 47.092146, 18.520376)
lat_step = 1.5
lon_step = 1.5
output_filename = "/home/aress/Documenti/tesi/progetto-esistente/ItalianSignNet/src/Dataminer/bounding_boxes.py"
output_html = "/home/aress/Documenti/tesi/progetto-esistente/ItalianSignNet/src/analizerTool/italy_grid.html"
bboxes_to_remove = [1,2,3,4,5,6,7,8,9,10,11,12,16,17,19,20,21,22,24,25,28,33,34,40,46,47,48,54,55,56,62,63,64]

def genera_grid():
    min_lat, min_lon, max_lat, max_lon = initial_bbox
    map_center = [(min_lat + max_lat)/2, (min_lon + max_lon)/2]
    
    mappa = folium.Map(
        location=map_center,
        zoom_start=6,
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='OpenStreetMap'
    )

    # Genera margini delle coordinate
    lat_edges = []
    current_lat = min_lat
    while current_lat < max_lat:
        lat_edges.append(current_lat)
        current_lat += lat_step
    lat_edges.append(max_lat)

    lon_edges = []
    current_lon = min_lon
    while current_lon < max_lon:
        lon_edges.append(current_lon)
        current_lon += lon_step
    lon_edges.append(max_lon)

    bounding_boxes = {}
    bbox_id = 1

    # Genera tutte le bbox
    for i in range(len(lat_edges)-1):
        for j in range(len(lon_edges)-1):
            bbox = (
                lat_edges[i],    # ll_lat
                lon_edges[j],    # ll_lon
                lat_edges[i+1],  # ur_lat
                lon_edges[j+1]  # ur_lon
            )
            current_id = str(bbox_id)  # ID come stringa
            
            if bbox_id in bboxes_to_remove:
                bbox_id += 1
                continue

            # Aggiungi alla mappa
            center_lat = (lat_edges[i] + lat_edges[i+1]) / 2
            center_lon = (lon_edges[j] + lon_edges[j+1]) / 2

            folium.Rectangle(
                bounds=[(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                color='#ff0000',
                fill=False,
                opacity=0.5,
                weight=1,
                popup=f"ID: {current_id}<br>BBOX: {bbox}"
            ).add_to(mappa)

            folium.Marker(
                location=[center_lat, center_lon],
                icon=DivIcon(
                    icon_size=(50,20),
                    icon_anchor=(25,10),
                    html=f'<div style="font-size: 10pt; color: red; background: white;">{current_id}</div>'
                )
            ).add_to(mappa)

            # Aggiungi al dizionario
            bounding_boxes[current_id] = {
                "ll_lat": bbox[0],
                "ll_lon": bbox[1],
                "ur_lat": bbox[2],
                "ur_lon": bbox[3]
            }

            bbox_id += 1

    # Scrivi file Python
    with open(output_filename, 'w') as f:
        f.write("# Bounding Box per Italia - Mapillary Tools\n\n")
        f.write(f"# Bbox rimosse: {bboxes_to_remove}\n")
        f.write("BOUNDING_BOXES = {\n")
        for bid, values in bounding_boxes.items():
            f.write(f'    "{bid}": {{\n')
            f.write(f'        "ll_lat": {values["ll_lat"]:.6f},\n')
            f.write(f'        "ll_lon": {values["ll_lon"]:.6f},\n')
            f.write(f'        "ur_lat": {values["ur_lat"]:.6f},\n')
            f.write(f'        "ur_lon": {values["ur_lon"]:.6f},\n')
            f.write('    },\n')
        f.write("}\n")
        f.write(f"\n# Totale Bounding Box generate: {len(bounding_boxes)}\n")

    # Salva mappa HTML
    mappa.save(output_html)
    print(f"File HTML generato: {output_html}")

if __name__ == "__main__":
    genera_grid()
    print(f"File Python generato: {output_filename}")
    print("Installare folium con: pip install folium")