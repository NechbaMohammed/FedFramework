import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
print("sys.path:", sys.path)
from fedml.datasets import FederatedDataset
print("fedml.datasets importé avec succès")
