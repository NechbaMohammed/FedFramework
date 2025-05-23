import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
# print("sys.path:", sys.path)  # Retiré, débogage

import ray
import os
import logging
import time
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Dict, Any
import torch
import numpy as np
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from flwr.simulation import start_simulation
from flwr.common import NDArrays, Scalar, Context
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server import Server, ServerConfig
from fedml.datasets import FederatedDataset
import hydra
from hydra.utils import instantiate
from fedml.algorithms.tamuna.model_utils import test
from fedml.algorithms.tamuna.strategy import TamunaStrategy
from fedml.algorithms.tamuna.server import TamunaServer
import importlib
from fedml.algorithms.tamuna.client import FlowerClientTamuna
importlib.reload(sys.modules['fedml.algorithms.tamuna.client'])
# print("Module client.py rechargé")  # Retiré, débogage
# print("FlowerClientTamuna importé depuis", FlowerClientTamuna.__module__, "à", FlowerClientTamuna.__code__.co_filename if hasattr(FlowerClientTamuna, '__code__') else "N/A")  # Retiré, débogage
# print("Importations terminées ##############################################################################################")  # Retiré, étape redondante

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de Ray avec runtime_env
runtime_env = {
    "py_modules": [os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'fedml'))],
    "excludes": ["__pycache__/*", "*.pyc", "*.pyo"]
}

# Initialisation de Ray avec runtime_env
start_time = time.time()
if ray.is_initialized():
    ray.shutdown()
ray.init(runtime_env=runtime_env)
logger.info(f"Ray initialisé en {time.time() - start_time:.2f} secondes")
# print("Ray bien initialisé ############################################################################################")  # Retiré, redondant avec logger

def tamuna_gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    client_cv_dir: str,
    epochs: int,
    learning_rate: float,
    model_cfg: DictConfig,
    device: torch.device,
    momentum: float = 0.9,
    weight_decay: float = 0.00001,
    compression_ratio: float = 0.5,
) -> Callable[[str], Any]:
    logger.info("Création de la fonction client TAMUNA")
    def client_fn(context: Context) -> Any:
        net = instantiate(model_cfg)
        net.to(device)
        cid = context.node_config["partition-id"]
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        logger.info(f"Client {cid} créé avec {len(trainloader.dataset)} échantillons d'entraînement")
        # print(f"Instanciation de FlowerClientTamuna pour client {cid} dans client_fn")  # Retiré, débogage
        client = FlowerClientTamuna(
            int(cid),
            net,
            trainloader,
            valloader,
            device,
            epochs,
            learning_rate,
            momentum,
            weight_decay,
            compression_ratio,
            save_dir=client_cv_dir
        )
        return client.to_client()
    return client_fn

def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    model_cfg: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    logger.info("Création de la fonction d'évaluation")
    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        net = instantiate(model_cfg)
        net.to(device)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        loss, accuracy = test(net, testloader, device)
        logger.info(f"Round {server_round} - Perte: {loss:.4f}, Précision: {accuracy:.4f}")
        return loss, {"test_accuracy": accuracy}
    return evaluate

def run_tamuna(
    data_config: DictConfig,
    model_cfg: DictConfig,
    backend_config: Dict,
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    learning_rate: float,
    model_dir: str,
    device: torch.device,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    fit_metrics_aggregation_fn=None,
    evaluate_metrics_aggregation_fn=None,
    min_fit_clients: Optional[int] = None,
    min_evaluate_clients: Optional[int] = None,
    min_available_clients: Optional[int] = None,
    compression_ratio: float = 0.5,
) -> Tuple[Any, ...]:
    logger.info("Démarrage de la simulation TAMUNA avec les paramètres fournis")
    logger.info(f"Configuration: {data_config}")

    client_cv_dir = model_dir

    logger.info(f"Utilisation du dispositif: {device}")
    logger.info(f"Nombre de clients: {num_clients}, Rounds: {num_rounds}, Époques: {num_epochs}")

    federated_dataset = FederatedDataset(data_config, num_clients=num_clients)
    trainloaders, valloaders, testloader = federated_dataset.get_dataloaders()
    logger.info(f"Nombre de trainloaders: {len(trainloaders)}, valloaders: {len(valloaders)}")

    tamuna_client_fn = tamuna_gen_client_fn(
        trainloaders=trainloaders,
        valloaders=valloaders,
        model_cfg=model_cfg,
        device=device,
        epochs=num_epochs,
        client_cv_dir=client_cv_dir,
        learning_rate=learning_rate,
        compression_ratio=compression_ratio,
    )
    tamuna_evaluate_fn = gen_evaluate_fn(
        testloader=testloader,
        device=device,
        model_cfg=model_cfg,
    )

    tamuna_strategy = TamunaStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        min_fit_clients=min_fit_clients or (num_clients // 2),
        min_evaluate_clients=min_evaluate_clients or num_clients,
        min_available_clients=min_available_clients or num_clients,
        evaluate_fn=tamuna_evaluate_fn,
    )
    net = instantiate(model_cfg)
    net.to(device)

    initial_parameters = [
        val.cpu().numpy() for _, val in net.state_dict().items()
    ]
    # print(f"Dimensions des paramètres initiaux du modèle: {[param.shape for param in initial_parameters]}")  # Retiré, débogage
    tamuna_server = TamunaServer(
        client_manager=SimpleClientManager(),
        strategy=tamuna_strategy,
        net=net,
    )

    logger.info("Lancement de la simulation...")
    tamuna_history = start_simulation(
        server=tamuna_server,
        client_fn=tamuna_client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=tamuna_strategy,
        client_resources=backend_config,
    )
    logger.info(f"Simulation terminée. Historique: {tamuna_history}")

    return (tamuna_history,)

if __name__ == "__main__":
    @hydra.main(config_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'conf')), config_name="config", version_base="1.1")
    def wrapped_main(cfg: DictConfig) -> None:
        run_tamuna(
            data_config=cfg.data_config,
            model_cfg=cfg.model_cfg,
            backend_config=cfg.backend_config,
            num_clients=10,
            num_rounds=3,
            num_epochs=5,
            learning_rate=0.01,
            model_dir="./tamuna_models",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=None,
            min_evaluate_clients=None,
            min_available_clients=None,
            compression_ratio=0.5,
        )
    wrapped_main()