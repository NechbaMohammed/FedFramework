from typing import Dict, Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flwr.common import NDArrays, Scalar
from flwr.client import NumPyClient
from .model_utils import TamunaOptimizer
import numpy as np
from PIL import Image

class FlowerClientTamuna(NumPyClient):
    """Client TAMUNA pour l'apprentissage fédéré avec compression des mises à jour locales."""

    def __init__(
        self,
        cid: int,
        net: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        compression_ratio: float,
        save_dir: str,
    ):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.compression_ratio = 1
        self.save_dir = save_dir
        self.control_vars = [
            torch.zeros_like(p.data, device='cpu', dtype=torch.float32, requires_grad=False).contiguous()
            for p in self.net.parameters()
        ]

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        parameters = []
        for idx, val in enumerate(self.net.state_dict().values()):
            if not isinstance(val, torch.Tensor):
                raise ValueError(f"Paramètre {idx} n'est pas un tenseur PyTorch, type reçu : {type(val)}")
            if val.requires_grad:
                val = val.detach()
            try:
                val = val.cpu().contiguous().numpy()
            except Exception as e:
                raise ValueError(f"Échec de la conversion du paramètre {idx} en NumPy : {str(e)}")
            if not isinstance(val, np.ndarray):
                raise ValueError(f"Paramètre {idx} après conversion n'est pas un numpy.ndarray, type reçu : {type(val)}")
            if np.any(np.isnan(val)) or np.any(np.isinf(val)):
                raise ValueError(f"Paramètre {idx} contient des valeurs NaN ou infinies après conversion")
            parameters.append(val)
        return parameters

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        num_model_layers = sum(1 for _ in self.net.parameters())
        if len(parameters) == num_model_layers:
            model_params = parameters
            control_vars = None
        elif len(parameters) == 2 * num_model_layers:
            split_point = len(parameters) // 2
            model_params = parameters[:split_point]
            control_vars = parameters[split_point:]
        else:
            raise ValueError(
                f"Longueur inattendue de parameters: {len(parameters)}. "
                f"Attendu {num_model_layers} (evaluate) ou {2 * num_model_layers} (fit)."
            )

        model_params_iter = iter(model_params)
        for idx, p in enumerate(self.net.parameters()):
            param = next(model_params_iter, None)
            if param is None:
                raise ValueError("Nombre insuffisant de paramètres pour le modèle.")
            if not isinstance(param, np.ndarray):
                raise ValueError(f"Paramètre {idx} attendu comme numpy.ndarray, reçu {type(param)}")
            if param.shape != p.data.shape:
                raise ValueError(f"Forme incompatible pour paramètre {idx} : attendu {p.data.shape}, reçu {param.shape}")
            p.data = torch.from_numpy(param).to(self.device)

        if control_vars is not None:
            if len(control_vars) != len(self.control_vars):
                raise ValueError(f"Incompatibilité des longueurs entre control_vars ({len(self.control_vars)}) et paramètres reçus ({len(control_vars)})")
            for idx, (cv, received_cv) in enumerate(zip(self.control_vars, control_vars)):
                if received_cv.shape != cv.shape:
                    raise ValueError(f"Forme incompatible pour control_vars[{idx}] : attendu {cv.shape}, reçu {received_cv.shape}")
                self.control_vars[idx] = torch.from_numpy(received_cv).to('cpu')

    def compress_update(self, updates):
        compressed_updates = []
        for update in updates:
            mask = np.random.binomial(1, self.compression_ratio, update.shape).astype(np.float32)
            compressed = update * mask
            compressed_updates.append(compressed)
        return compressed_updates

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        
        optimizer = TamunaOptimizer(
            self.net.parameters(),
            step_size=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.net.train()
        num_examples = 0

        initial_params = [p.detach().cpu().numpy().copy() for p in self.net.parameters()]
        server_cv = [torch.from_numpy(p).to(self.device) for p in parameters[len(parameters)//2:]]
        client_cv = [cv.to(self.device) for cv in self.control_vars]

        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.net(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step_custom(server_cv=server_cv, client_cv=client_cv)
                num_examples += target.size(0)

        final_params = [p.detach().cpu().numpy().copy() for p in self.net.parameters()]
        param_updates = [(new - old) for old, new in zip(initial_params, final_params)]
        cv_params = parameters[len(parameters)//2:]
        cv_update = [
            ((cv - torch.from_numpy(p).to('cpu')) / (self.epochs * len(self.trainloader.dataset) / self.trainloader.batch_size)).cpu().numpy()
            for cv, p in zip(self.control_vars, cv_params)
        ]

        for i, pu in enumerate(param_updates):
            self.control_vars[i] += torch.from_numpy(pu).to('cpu') / (self.epochs * len(self.trainloader.dataset) / self.trainloader.batch_size)

        compressed_updates = self.compress_update(param_updates)
        compressed_cv_update = self.compress_update(cv_update)
        combined_updates = compressed_updates + compressed_cv_update

        return combined_updates, num_examples, {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        self.net.eval()
        loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for data, target in self.valloader:
                data, target = data.to(self.device), target.to(self.device)
                if data.dim() == 3 and data.size(0) == 1:
                    data = data.squeeze(0)
                outputs = self.net(data)
                loss += nn.CrossEntropyLoss()(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        loss /= len(self.valloader)
        accuracy = correct / total
        num_examples = total
        return loss, num_examples, {"accuracy": accuracy}