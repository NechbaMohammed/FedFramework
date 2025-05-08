from typing import List, Tuple
import torch
import torch.nn as nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
import numpy as np

class TamunaOptimizer(SGD):
    """Optimiseur TAMUNA pour appliquer les corrections des variables de contrôle."""

    def __init__(self, params, step_size, momentum, weight_decay):
        super().__init__(
            params, lr=step_size, momentum=momentum, weight_decay=weight_decay
        )

    def step_custom(self, server_cv, client_cv):
        """Applique une étape personnalisée avec correction des variables de contrôle."""
        self.step()
        for group in self.param_groups:
            for par, s_cv, c_cv in zip(group["params"], server_cv, client_cv):
                par.data.add_(s_cv - c_cv, alpha=-group["lr"])

def train_tamuna(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    server_cv: List[torch.Tensor],
    client_cv: List[torch.Tensor],
    compression_ratio: float,
) -> None:
    """Entraîne le réseau avec TAMUNA, incluant la compression des mises à jour."""
    criterion = nn.CrossEntropyLoss()
    optimizer = TamunaOptimizer(
        net.parameters(), learning_rate, momentum, weight_decay
    )
    net.train()
    for _ in range(epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step_custom(server_cv, client_cv)

def compute_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Calcule la précision et la perte."""
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    loss = loss / total
    acc = correct / total
    return loss, acc

def test(net: nn.Module, test_dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Évalue le modèle sur l'ensemble de test."""
    net.to(device)
    loss, test_acc = compute_accuracy(net, test_dataloader, device=device)
    net.to("cpu")
    return loss, test_acc