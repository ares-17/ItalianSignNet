import os
import json
from dotenv import load_dotenv
from pathlib import Path
from collections import Counter

load_dotenv()
ALL_ANNOTATIONS_FOLDER = Path(os.getenv("ALL_ANNOTATIONS_FOLDER")) / "annotations_image"
locations = []

for filename in os.listdir(ALL_ANNOTATIONS_FOLDER):
    if filename.endswith('.json'):
        filepath = os.path.join(ALL_ANNOTATIONS_FOLDER, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
            location = data["image"].get("geographic_location")
            if location:
                locations.append(location)

counts = Counter(locations)
print("Cartelli per area geografica:")
print("Sud:", counts.get("sud", 0))
print("Centro:", counts.get("centre", 0))
print("Nord:", counts.get("nord", 0))
