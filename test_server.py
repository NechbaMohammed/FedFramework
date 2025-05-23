import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
print("sys.path:", sys.path)
from fedml.algorithms.tamuna.server import TamunaServer
print("TamunaServer importé avec succès")