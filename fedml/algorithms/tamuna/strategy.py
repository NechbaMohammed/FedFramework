from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import Scalar
import torch
import flwr as fl
from collections import OrderedDict
import logging
from logging import DEBUG, INFO, WARNING
from flwr.common.logger import log
from functools import reduce
import numpy as np
from flwr.common import Parameters, Scalar
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.common.typing import FitRes
from flwr.server.strategy.aggregate import aggregate
from flwr.server.strategy import FedAvg

class TamunaStrategy(FedAvg):
    """Stratégie TAMUNA basée sur FedAvg avec compression et participation partielle."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Agrège les résultats d'apprentissage en utilisant une moyenne pondérée."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Extraire les paramètres combinés et le nombre d'exemples
        combined_parameters_all_updates = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        len_combined_parameter = len(combined_parameters_all_updates[0])
        num_examples_all_updates = [fit_res.num_examples for _, fit_res in results]

        # Vérifier la cohérence des longueurs des paramètres combinés
        if not all(len(update) == len_combined_parameter for update in combined_parameters_all_updates):
            log(WARNING, f"Incohérence dans la longueur des paramètres combinés au round {server_round}")
            return None, {}

        # Vérifier que la longueur est paire (paramètres + cv_update)
        if len_combined_parameter % 2 != 0:
            log(WARNING, f"Longueur impaire des paramètres combinés ({len_combined_parameter}) au round {server_round}")
            return None, {}

        # Séparer les paramètres du modèle et les mises à jour des variables de contrôle
        num_layers = len_combined_parameter // 2
        weights_results = [
            (update[:num_layers], num_examples)
            for update, num_examples in zip(combined_parameters_all_updates, num_examples_all_updates)
        ]
        parameters_aggregated = aggregate(weights_results)

        client_cv_updates_and_num_examples = [
            (update[num_layers:], num_examples)
            for update, num_examples in zip(combined_parameters_all_updates, num_examples_all_updates)
        ]

        # Vérifier la cohérence des dimensions des cv_update
        cv_update_shapes = [tuple(np.array(update).shape for update in client_cvs) for client_cvs, _ in client_cv_updates_and_num_examples]
        if len(set(cv_update_shapes)) > 1:
            log(WARNING, f"Dimensions incohérentes des cv_update: {cv_update_shapes} au round {server_round}")
            return None, {}

        # Agréger les mises à jour des variables de contrôle
        aggregated_cv_update = aggregate(client_cv_updates_and_num_examples)

        # Agréger les métriques personnalisées si une fonction est fournie
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "Aucune fonction fit_metrics_aggregation_fn fournie")

        # Combiner les paramètres agrégés et les mises à jour des variables de contrôle
        combined_aggregated = parameters_aggregated + aggregated_cv_update
        return ndarrays_to_parameters(combined_aggregated), metrics_aggregated