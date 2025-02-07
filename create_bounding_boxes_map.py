import folium
import json
from src.Dataminer.bounding_boxes import BOUNDING_BOXES

mappa = folium.Map(location=[42.0, 12.5], zoom_start=6)

for regione, dati in BOUNDING_BOXES.items():
    box = [
        [dati["ll_lat"], dati["ll_lon"]],  # Lower left
        [dati["ur_lat"], dati["ll_lon"]],  # Upper left
        [dati["ur_lat"], dati["ur_lon"]],  # Upper right
        [dati["ll_lat"], dati["ur_lon"]],  # Lower right
        [dati["ll_lat"], dati["ll_lon"]]   # Closing the box
    ]    
    folium.PolyLine(box, color="blue", weight=2.5, popup=dati["nome"]).add_to(mappa)

mappa.save("public/bounding_boxes_italia.html")
