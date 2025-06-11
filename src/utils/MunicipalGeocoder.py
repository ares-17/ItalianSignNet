import json
from shapely.geometry import Point, shape
from shapely.prepared import prep
from typing import Optional, Dict, Any, List, Tuple
import logging

class MunicipalGeocoder:
    """
    Class for geocoding geographic coordinates and obtaining the ISTAT code of the municipality.
    Uses a hierarchical approach: first identifies the region, then the municipality.
    """
    
    def __init__(self, regions_file: str, municipalities_file: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the geocoder by loading GeoJSON files.
        
        Args:
            regions_file: Path to the regions GeoJSON file
            municipalities_file: Path to the municipalities GeoJSON file
            logger: Logger instance to use. If None, creates a default logger
        """
        self.logger = logger or self._create_default_logger()
        
        self.regions = {}
        self.municipalities_per_region = {}
        self.prepared_regions = {}
        self.prepared_municipalities = {}
        
        self._load_regions(regions_file)
        self._load_municipalities(municipalities_file)
    
    def _create_default_logger(self) -> logging.Logger:
        """Create a default logger if none is provided."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_regions(self, regions_file: str):
        """Load region polygons from GeoJSON file."""
        try:
            with open(regions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for feature in data['features']:
                # Extract region information
                properties = feature['properties']
                geometry = feature['geometry']
                
                # Use region code as key (adapt based on your file)
                # Assume there's a field similar to 'reg_istat_code' or 'cod_reg'
                reg_code = (properties.get('reg_istat_code') or 
                           properties.get('cod_reg') or 
                           properties.get('ISTAT_COD_REG'))
                reg_name = (properties.get('reg_name') or 
                           properties.get('nome_reg') or 
                           properties.get('NOME_REG'))
                
                if reg_code:
                    # Create Shapely object and prepared version for fast searches
                    geom = shape(geometry)
                    self.regions[reg_code] = {
                        'name': reg_name,
                        'geometry': geom,
                        'properties': properties
                    }
                    self.prepared_regions[reg_code] = prep(geom)
            
            self.logger.info(f"Loaded {len(self.regions)} regions")
            
        except Exception as e:
            self.logger.error(f"Error loading regions: {e}")
            raise
    
    def _load_municipalities(self, municipalities_file: str):
        """Load municipality polygons from GeoJSON file, organizing them by region."""
        try:
            with open(municipalities_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for feature in data['features']:
                properties = feature['properties']
                geometry = feature['geometry']
                
                # Extract necessary codes
                com_istat_code = properties.get('com_istat_code')
                reg_istat_code = properties.get('reg_istat_code')
                
                if com_istat_code and reg_istat_code:
                    # Organize municipalities by region
                    if reg_istat_code not in self.municipalities_per_region:
                        self.municipalities_per_region[reg_istat_code] = {}
                        self.prepared_municipalities[reg_istat_code] = {}
                    
                    # Create Shapely object and prepared version
                    geom = shape(geometry)
                    self.municipalities_per_region[reg_istat_code][com_istat_code] = {
                        'geometry': geom,
                        'properties': properties
                    }
                    self.prepared_municipalities[reg_istat_code][com_istat_code] = prep(geom)
            
            total_municipalities = sum(len(municipalities) for municipalities in self.municipalities_per_region.values())
            self.logger.info(f"Loaded {total_municipalities} municipalities in {len(self.municipalities_per_region)} regions")
            
        except Exception as e:
            self.logger.error(f"Error loading municipalities: {e}")
            raise
    
    def _find_region(self, point: Point) -> Optional[str]:
        """
        Find the region that contains the given point.
        
        Args:
            point: Shapely Point with coordinates
            
        Returns:
            ISTAT code of the region or None if not found
        """
        for reg_code, geom_prep in self.prepared_regions.items():
            if geom_prep.contains(point):
                return reg_code
        return None
    
    def _find_municipality_in_region(self, point: Point, reg_code: str) -> Optional[str]:
        """
        Find the municipality that contains the point within a specific region.
        
        Args:
            point: Shapely Point with coordinates
            reg_code: ISTAT code of the region
            
        Returns:
            ISTAT code of the municipality or None if not found
        """
        if reg_code not in self.prepared_municipalities:
            return None
        
        for com_code, geom_prep in self.prepared_municipalities[reg_code].items():
            if geom_prep.contains(point):
                return com_code
        return None
    
    def geocode(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """
        Geocode a pair of coordinates returning the ISTAT code of the municipality.
        
        Args:
            latitude: Latitude in decimal degrees (WGS84)
            longitude: Longitude in decimal degrees (WGS84)
            
        Returns:
            Dictionary with the municipality ISTAT code,
            or None if the point doesn't fall within any municipality
        """
        try:
            # Create the point (note: Shapely uses (lon, lat))
            point = Point(longitude, latitude)
            
            # First find the region
            reg_code = self._find_region(point)
            if not reg_code:
                self.logger.warning(f"No region found for coordinates ({latitude}, {longitude})")
                return None
            
            # Then find the municipality within the region
            com_code = self._find_municipality_in_region(point, reg_code)
            if not com_code:
                self.logger.warning(f"No municipality found for coordinates ({latitude}, {longitude}) in region {reg_code}")
                return None
             
            return {
                'com_istat_code': com_code
            }
            
        except Exception as e:
            self.logger.error(f"Error geocoding ({latitude}, {longitude}): {e}")
            return None
    
    def geocode_batch(self, coordinates_list: List[Tuple[float, float]]) -> List[Optional[Dict[str, Any]]]:
        """
        Geocode a list of coordinates.
        
        Args:
            coordinates_list: List of tuples (latitude, longitude)
            
        Returns:
            List of results (dictionaries or None for non-geocoded coordinates)
        """
        results = []
        for lat, lon in coordinates_list:
            result = self.geocode(lat, lon)
            results.append(result)
        
        return results