"""
Import helmpy by absolute path and define TEST_PATH (path inside de main folder of the repository)
"""

import sys
from pathlib import Path

HELMPY_PATH = Path(__file__).parents[1]
sys.path.append(str(HELMPY_PATH))

import helmpy
