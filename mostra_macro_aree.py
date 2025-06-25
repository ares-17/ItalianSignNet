# generate_macro_area_map.py

import json
import os
import sys
import folium
from dotenv import load_dotenv
from shapely.geometry import shape

# Setup path per import personalizzati
utils_path = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(utils_path)

from utils.RegionGeocoder import RegionGeocoder

# Configurazione base
load_dotenv()
BASE_DIR = os.getenv("BASE_DIR", os.path.dirname(__file__))
REGIONS_FILE = os.path.join(BASE_DIR, 'src', 'resources', 'limits_IT_regions.geojson')

COLOR_MAP = {
    'nord': '#1f77b4',
    'centre': '#2ca02c',
    'sud': '#ff7f0e'
}

REGION_NAME_NORMALIZATION = {
    "Valle d'Aosta": "Regione Valle d'Aosta/Vallée d'Aoste",
    "Vallée d'Aoste": "Regione Valle d'Aosta/Vallée d'Aoste",
    "Valle d'Aosta/Vallée d'Aoste": "Regione Valle d'Aosta/Vallée d'Aoste",
    "Trentino-Alto Adige/Südtirol": "Trentino-Alto Adige",
    "Provincia Autonoma Trento": "Trentino-Alto Adige",
    "Provincia Autonoma Bolzano/Bozen": "Trentino-Alto Adige"
}

def get_centroid(feature):
    try:
        geom = shape(feature['geometry'])
        return [geom.centroid.y, geom.centroid.x]
    except Exception as e:
        print(f"Errore nel calcolo del baricentro: {e}")
        return [45.0, 10.0]  # Fallback generico

def load_and_enrich_geojson(geojson_path):
    geocoder = RegionGeocoder(geojson_path)
    with open(geojson_path, encoding='utf-8') as f:
        geojson_data = json.load(f)

    valid_features = []
    for feature in geojson_data['features']:
        props = feature.get("properties", {})
        reg_code = str(props.get("reg_istat_code") or props.get("cod_reg") or props.get("ISTAT_COD_REG"))
        reg_name = props.get("reg_name")

        if reg_code not in geocoder.regions:
            print(f"[!] Codice regione '{reg_code}' non trovato per '{reg_name}' → esclusa")
            continue

        region_data = geocoder.regions[reg_code]
        actual_name = region_data['name']
        normalized_name = REGION_NAME_NORMALIZATION.get(actual_name, actual_name)
        macro_area = geocoder.MACRO_AREA_MAP.get(normalized_name)

        if not macro_area:
            print(f"[!] Regione '{actual_name}' non ha macro-area → fallback a 'sud'")
            macro_area = 'sud'

        feature['properties'].update({
            'macro_area': macro_area,
            'fillColor': COLOR_MAP.get(macro_area, '#cccccc'),
            'fillOpacity': 0.7,
            'normalized_name': normalized_name
        })

        valid_features.append(feature)

    geojson_data['features'] = valid_features
    return geojson_data


def create_map(geojson_data):
    m = folium.Map(location=[42.5, 12.5], zoom_start=6, tiles='cartodbpositron', control_scale=True)

    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'fillColor': feature['properties']['fillColor'],
            'color': '#333',
            'weight': 1.5,
            'fillOpacity': feature['properties'].get('fillOpacity', 0.6),
        },
        highlight_function=lambda feature: {
            'weight': 3,
            'color': '#000',
            'fillOpacity': 0.9,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['reg_name', 'macro_area'],
            aliases=['Regione:', 'Macro-area:'],
            sticky=True
        )
    ).add_to(m)

    legend_html = '''
    <div style="
        position: fixed;
        bottom: 50px; left: 50px;
        width: 180px;
        border: 2px solid grey;
        z-index: 9999;
        font-size: 14px;
        background: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 0 5px rgba(0,0,0,0.3);
    ">
        <div style="font-weight: bold; margin-bottom: 5px;">Macro-aree italiane</div>
        <div><i style="background:#1f77b4; width:15px; height:15px; display:inline-block;"></i> Nord</div>
        <div><i style="background:#2ca02c; width:15px; height:15px; display:inline-block;"></i> Centro</div>
        <div><i style="background:#ff7f0e; width:15px; height:15px; display:inline-block;"></i> Sud/Isole</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

if __name__ == "__main__":
    try:
        print(">> Caricamento e preparazione dei dati geografici...")
        enriched_data = load_and_enrich_geojson(REGIONS_FILE)

        print(">> Generazione della mappa...")
        italy_map = create_map(enriched_data)

        output_html = "italy_macro_areas.html"
        italy_map.save(output_html)
        print(f">> ✅ Mappa salvata in: {output_html}")
    except Exception as e:
        print(f">> ❌ Errore durante l'esecuzione: {e}")
