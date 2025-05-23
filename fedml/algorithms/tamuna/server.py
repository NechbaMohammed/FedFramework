from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.server.server import FitResultsAndFailures, Server, fit_clients
from flwr.common import Parameters, Scalar
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.common.logger import log
import logging
from logging import INFO, DEBUG
import torch
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import GetParametersIns
from flwr.server.strategy import Strategy

class TamunaServer(Server):
    """Serveur TAMUNA pour l'apprentissage fédéré avec compression et participation partielle."""

    def __init__(
        self,
        strategy: Strategy,
        net: torch.nn.Module,
        client_manager: Optional[ClientManager] = None,
    ):        
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)
        
        # Initialiser les paramètres du modèle et les variables de contrôle du serveur
        self.model_params = net
        model_ndarrays = [val.cpu().numpy() for val in self.model_params.state_dict().values()]
        self.parameters = ndarrays_to_parameters(model_ndarrays)
        self.server_cv = [
            torch.zeros_like(torch.Tensor(param)) 
            for param in model_ndarrays
        ]

    def _get_initial_parameters(self, timeout: Optional[float], **kwargs) -> Parameters:
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Utilisation des paramètres initiaux fournis par la stratégie")
            return parameters
        
        log(INFO, "Demande des paramètres initiaux à un client aléatoire")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout, group_id="default_group")
        
        log(INFO, "Paramètres initiaux reçus d'un client aléatoire")
        return get_parameters_res.parameters

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Effectuer un round d'apprentissage fédéré avec TAMUNA."""
        # Configurer les instructions pour les clients avec les paramètres du modèle et les variables de contrôle
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=update_parameters_with_cv(self.parameters, self.server_cv),
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, f"fit_round {server_round}: aucun client sélectionné, annulation")
            return None
        log(
            DEBUG,
            f"fit_round {server_round}: la stratégie a échantillonné {len(client_instructions)} clients (sur {self._client_manager.num_available()})",
        )

        # Collecter les résultats d'apprentissage des clients participants
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=str(server_round)
        )
        log(
            DEBUG,
            f"fit_round {server_round} a reçu {len(results)} résultats et {len(failures)} échecs",
        )

        # Agréger les résultats d'apprentissage
        aggregated_result: Tuple[Optional[Parameters], Dict[str, Scalar]] = (
            self.strategy.aggregate_fit(server_round, results, failures)
        )
        if aggregated_result[0] is None:
            return None

        aggregated_result_arrays_combined = parameters_to_ndarrays(
            aggregated_result[0]
        )
        split_point = len(aggregated_result_arrays_combined) // 2
        aggregated_parameters = aggregated_result_arrays_combined[:split_point]
        aggregated_cv_update = aggregated_result_arrays_combined[split_point:]

        # Vérifier si les longueurs correspondent
        if len(self.server_cv) != len(aggregated_cv_update):
            log(logging.ERROR, f"Incompatibilité des longueurs entre server_cv ({len(self.server_cv)}) et aggregated_cv_update ({len(aggregated_cv_update)})")
            return None

        # Vérifier les dimensions de chaque couche
        for i, (server_cv_layer, cv_update_layer) in enumerate(zip(self.server_cv, aggregated_cv_update)):
            if server_cv_layer.shape != cv_update_layer.shape:
                log(logging.ERROR, f"Incompatibilité des dimensions à la couche {i}: server_cv {server_cv_layer.shape} vs aggregated_cv_update {cv_update_layer.shape}")
                return None

        # Convertir server_cv en ndarrays
        server_cv_np = [cv.numpy() for cv in self.server_cv]
        # Mettre à jour server_cv
        total_clients = len(self._client_manager.all())
        cv_multiplier = len(results) / total_clients
        self.server_cv = [
            torch.from_numpy(cv + cv_multiplier * aggregated_cv_update[i])
            for i, cv in enumerate(server_cv_np)
        ]

        # Mettre à jour les paramètres du modèle
        curr_params = parameters_to_ndarrays(self.parameters)
        updated_params = [
            x + aggregated_parameters[i] for i, x in enumerate(curr_params)
        ]
        parameters_updated = ndarrays_to_parameters(updated_params)

        # Métriques
        metrics_aggregated = aggregated_result[1]
        return parameters_updated, metrics_aggregated, (results, failures)

def update_parameters_with_cv(
    parameters: Parameters, s_cv: List[torch.Tensor]
) -> Parameters:
    """Ajouter les variables de contrôle du serveur aux paramètres du modèle."""
    parameters_np = parameters_to_ndarrays(parameters)
    cv_np = [cv.numpy() for cv in s_cv]
    parameters_np.extend(cv_np)
    return ndarrays_to_parameters(parameters_np)