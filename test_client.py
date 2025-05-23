# E:\FedFramework\test_client.py
import sys
print("sys.path:", sys.path)
from fedml.algorithms.tamuna.client import FlowerClientTamuna
print("Chargement de FlowerClientTamuna depuis", FlowerClientTamuna.__module__, "à", FlowerClientTamuna.__code__.co_filename if hasattr(FlowerClientTamuna, '__code__' ) else "N/A")
print("Méthode evaluate définie:", hasattr(FlowerClientTamuna, 'evaluate'))