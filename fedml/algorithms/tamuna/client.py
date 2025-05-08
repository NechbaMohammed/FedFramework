import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from collections import OrderedDict
import flwr as fl
from flwr.common import Scalar
from fedml.algorithms.tamuna.model_utils import train_tamuna, test
import numpy as np
import os

class FlowerClientTamuna(fl.client.NumPyClient):
    """Client TAMUNA pour l'apprentissage fédéré."""

    def __init__(
        self,
        cid: int,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        compression_ratio: float,
        save_dir: str = "",
    ) -> None:
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.compression_ratio = compression_ratio
        self.client_cv = [torch.zeros_like(p) for p in self.net.parameters()]
        if save_dir == "":
            save_dir = "client_cvs"
        self.dir = save_dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        model_weights = parameters[:len(parameters)//2]
        server_cv = parameters[len(parameters)//2:]
        self.set_parameters(model_weights)
        server_cv = [torch.Tensor(cv) for cv in server_cv]
        train_tamuna(
            self.net,
            self.trainloader,
            self.device,
            self.num_epochs,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
            server_cv,
            self.client_cv,
            self.compression_ratio,
        )
        updated_weights = self.get_parameters(config={})
        delta_weights = [np.subtract(updated, initial) for updated, initial in zip(updated_weights, model_weights)]
        cv_updates = [
            (1.0 / (self.learning_rate * self.num_epochs * len(self.trainloader))) * (updated - initial)
            for updated, initial in zip(updated_weights, model_weights)
        ]
        self.client_cv = [
            c_i_j - c_j + cv_update
            for c_i_j, c_j, cv_update in zip(self.client_cv, server_cv, cv_updates)
        ]
        torch.save(self.client_cv, f"{self.dir}/client_cv_{self.cid}.pt")
        return delta_weights + cv_updates, len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        loss, acc = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}