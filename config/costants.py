from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(os.getenv('BASE_DIR'))

__all__ = ['BASE_DIR']