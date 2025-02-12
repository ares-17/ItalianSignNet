import folium
import os
import sys

# Aggiungi il percorso della cartella "utils" al PYTHONPATH
utils_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(utils_path)

from dataminer.bounding_boxes import BOUNDING_BOXES

mappa = folium.Map(location=[42.0, 12.5], zoom_start=6)

for regione, dati in BOUNDING_BOXES.items():
    box = [
        [dati["ll_lat"], dati["ll_lon"]],  # Lower left
        [dati["ur_lat"], dati["ll_lon"]],  # Upper left
        [dati["ur_lat"], dati["ur_lon"]],  # Upper right
        [dati["ll_lat"], dati["ur_lon"]],  # Lower right
        [dati["ll_lat"], dati["ll_lon"]]   # Closing the box
    ]
    
    folium.PolyLine(box, color="blue", weight=2.5).add_to(mappa)

    center_lat = (dati["ll_lat"] + dati["ur_lat"]) / 2
    center_lon = (dati["ll_lon"] + dati["ur_lon"]) / 2

    folium.Marker(
        location=[center_lat, center_lon],
        icon=folium.DivIcon(
            icon_size=(50, 20),
            icon_anchor=(25, 10),
            html=f'<div style="font-size: 12pt; color: blue; border-radius: 3px; padding: 2px;">{regione}</div>'
        )
    ).add_to(mappa)

mappa.save("public/bounding_boxes_italia.html")
print("Mappa salvata in public/bounding_boxes_italia.html")