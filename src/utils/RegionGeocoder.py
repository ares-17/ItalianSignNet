import json
from shapely.geometry import Point, shape
from shapely.prepared import prep
from typing import Optional, Dict, Any, List, Tuple
import logging

class RegionGeocoder:
    """
    Classe per geocodificare coordinate e ottenere:
      - nome della regione
      - macro-area geografica (Nord/Centro/Sud)
      - codice ISTAT della regione
    La ripartizione delle regioni in macro-aree segue quella ufficiale disponibile al link:
    https://www.agenziacoesione.gov.it/sistema-conti-pubblici-territoriali/il-sistema-cpt/metodologia/
    """

    """
    I nomi seguono la convenzione dei dati censiti in:
    https://github.com/openpolis/geojson-italy/blob/master/geojson/limits_IT_regions.geojson
    """
    MACRO_AREA_MAP = {
        # Nord
        'Piemonte':'nord', "Valle d'Aosta/Vallée d'Aoste":'nord',
        'Liguria':'nord', 'Lombardia':'nord',
        'Trentino-Alto Adige/Südtirol':'nord', 'Veneto':'nord',
        'Friuli-Venezia Giulia':'nord', 'Emilia-Romagna':'nord',
        # Centro
        'Toscana':'centre', 'Marche':'centre',
        'Umbria':'centre', 'Lazio':'centre',
        # Sud
        'Abruzzo':'sud', 'Molise':'sud',
        'Campania':'sud', 'Puglia':'sud',
        'Basilicata':'sud', 'Calabria':'sud',
        # Isole (considerate Sud)
        'Sicilia':'sud', 'Sardegna':'sud'
    }

    def __init__(self, regions_file: str, logger: Optional[logging.Logger] = None):
        self.logger = logger or self._create_logger()
        self.regions = {}  # ISTAT code → {'name', 'geometry', 'prepared'}
        self._load_regions(regions_file)

    def _create_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            logger.addHandler(h)
            logger.setLevel(logging.INFO)
        return logger

    def _load_regions(self, path: str):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        for feat in data['features']:
            props = feat['properties']
            geom = shape(feat['geometry'])
            reg_code = props.get('reg_istat_code') or props.get('cod_reg') or props.get('ISTAT_COD_REG')
            reg_name = props.get('reg_name') or props.get('nome_reg') or props.get('NOME_REG')
            if reg_code and reg_name:
                self.regions[reg_code] = {
                    'name': reg_name,
                    'geometry': geom,
                    'prepared': prep(geom)
                }
        self.logger.info(f"Loaded {len(self.regions)} regions")

    def geocode(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """
        Geocode coordinate → ritorna {
            'reg_istat_code', 'reg_name', 'macro_area'
        } oppure, se non matcha, assegna la regione più vicina.
        """
        pt = Point(longitude, latitude)
        closest_region = None
        min_dist = float('inf')
        closest_code = None

        for code, region in self.regions.items():
            geom = region['geometry']
            if region['prepared'].intersects(pt):
                name = region['name']
                macro = self.MACRO_AREA_MAP.get(name)
                if macro is None:
                    self.logger.warning(f"Regione {name} senza macro-area definita")
                return {
                    'reg_istat_code': code,
                    'reg_name': name,
                    'macro_area': macro
                }

            # Calcolo distanza al poligono
            dist = geom.distance(pt)
            if dist < min_dist:
                min_dist = dist
                closest_region = region
                closest_code = code

        # Fallback: regione più vicina
        if closest_region:
            name = closest_region['name']
            macro = self.MACRO_AREA_MAP.get(name)
            self.logger.warning(
                f"Coordinate ({latitude},{longitude}) non interne a nessuna regione. "
                f"Assegnata regione più vicina: '{name}' (distanza {min_dist:.6f})"
            )
            return {
                'reg_istat_code': closest_code,
                'reg_name': name,
                'macro_area': macro
            }

        # Se non trova nemmeno la più vicina (cosa impossibile se dati validi)
        self.logger.error(f"Coordinate ({latitude},{longitude}) non assegnabili a nessuna regione nemmeno per prossimità.")
        return None


    def geocode_batch(self, coords: List[Tuple[float, float]]) -> List[Optional[Dict[str, Any]]]:
        return [self.geocode(lat, lon) for lat, lon in coords]
