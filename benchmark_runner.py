import os
import subprocess
import time
from collections import deque
from typing import List

MAX_PROCESSES_AT_ONCE = 1  # Number of processes to run at once

rounds = 10  # Number of rounds (modifié de 1 à 10)
epochs = "1 2"  # List Number of local epochs (inchangé)
runs = 1  # Number of runs to execute (inchangé)
num_clients = 10  # for all partitioning except cifar100 (inchangé)
alpha = 0.5  # for dirichlet partitioning
segma = 0.1  # for noise partitioning
similarity = 0.5  # for iid_noniid partitioning
labels_per_client = "1 2 3"  # for label_quantity partitioning
learning_rate = 0.01 

# Define the methods and datasets to run
methods = ["tamuna"]  # Suppression de "scaffold", ne garder que "tamuna"
datasets = ["mnist", "fmnist", "cifar10"]
partitioning = {
    "mnist": ["label_quantity", "dirichlet", "iid_noniid", "noise", "iid"],
    "fmnist": ["label_quantity", "dirichlet", "iid_noniid", "noise", "iid"],
    "cifar10": ["label_quantity", "dirichlet", "iid_noniid", "noise", "iid"],
}

# Boucle pour exécuter les commandes
commands: deque = deque()
for method in methods:
    for dataset in datasets:
        for part in partitioning[dataset]:
            cmd = f"python main.py --method {method} --dataset {dataset} --partitioning {part} --num_clients {num_clients} --learning_rate {learning_rate} --rounds {rounds} --epochs {epochs} --runs {runs}"
            
            if part == "label_quantity":
                cmd += f" --labels_per_client {labels_per_client}"
            elif part == "dirichlet":
                cmd += f" --alpha {alpha}"
            elif part == "iid_noniid":
                cmd += f" --similarity {similarity}"
            elif part == "noise":
                cmd += f" --segma {segma}"
            elif part == "iid":
                pass  # No additional parameters needed for iid
            commands.append(cmd)

# run max_processes_at_once processes at once with 10 second sleep interval
# in between those processes until all commands are done

processes: List = []
while len(commands) > 0:
    while len(processes) < MAX_PROCESSES_AT_ONCE and len(commands) > 0:
        cmd = commands.popleft()
        print(cmd)
        processes.append(subprocess.Popen(cmd, shell=True))
        # sleep for 10 seconds to give the process time to start
        time.sleep(10)
    for p in processes:
        if p.poll() is not None:
            processes.remove(p)