# Federated Learning Framework

## Quick Start Tutorials
#### Dataset Usage Examples
Below are examples demonstrating how to use the FLBenchmark dataset module.
```python
from fedbench.datasets import FederatedDataset
from omegaconf import DictConfig

# Example 1: MNIST Dataset

data_config = DictConfig({"name": "mnist", "partitioning": "iid", "batch_size": 64})

federated_dataset = FederatedDataset(data_config, num_clients=1)
trainloaders, valloaders, testloader = federated_dataset.get_dataloaders()
federated_dataset.print_dataset_stats()
```
```
Number of training instances: 60000
Number of test instances: 10000
Number of features: 784
Number of classes: 10
```

```python
# Example 5: Label Quantity Partitiin

data_config = DictConfig({"name": "mnist", "partitioning": "label_quantity","labels_per_client":2, "batch_size": 64})

federated_dataset = FederatedDataset(data_config, num_clients=10)
trainloaders, valloaders, testloader = federated_dataset.get_dataloaders()
```
```python
# Example 6: Dirichlet Partitiin

data_config = DictConfig({"name": "mnist", "partitioning": "dirichlet","alpha":0.5, "batch_size": 64})

federated_dataset = FederatedDataset(data_config, num_clients=10)
trainloaders, valloaders, testloader = federated_dataset.get_dataloaders()
```
```python
# Example 8: Feature Distribution Partitiin

data_config = DictConfig({"name": "mnist", "partitioning": "noise","segma":0.1, "batch_size": 64})

federated_dataset = FederatedDataset(data_config, num_clients=10)
trainloaders, valloaders, testloader = federated_dataset.get_dataloaders()
```
### Strategic Algorithm 
##### Scaffold on CIFAR10
```python

from fedbench.algorithms.scaffold.simulation import run_scaffold
from omegaconf import OmegaConf,DictConfig
import torch

backend_config = {
    "num_cpus": 1,
    "num_gpus": 0
}
model_cfg = OmegaConf.create({
            "_target_": "fedml.models.CNN" ,
            "input_dim": 400,
            "hidden_dims": [120, 84],
            "num_classes": 10,
        })
data_config = DictConfig({
    "name": "cifar10",
    "partitioning": "iid_noniid",
    "similarity":0.5,
    "batch_size": 32,
})


history = run_scaffold(
    data_config=data_config,
    model_cfg=model_cfg,
    backend_config=backend_config,
    num_clients=15,
    num_rounds=25,
    num_epochs=20,
    learning_rate=0.001,
    model_dir="weights",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```
