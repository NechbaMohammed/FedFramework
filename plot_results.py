import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns  # Importer Seaborn pour le style
import argparse
import numpy as np

# Seaborn est importé, il applique automatiquement un style élégant à Matplotlib
# Si Seaborn n'est pas installé, commentez la ligne ci-dessus et décommentez celle-ci :
# plt.style.use('ggplot')  # Style alternatif intégré à Matplotlib

# Définir les constantes
BASE_PATH = os.path.abspath("results\\experiments\\tamuna")  # Convertir en chemin absolu
DATASETS = ["mnist", "fmnist", "cifar10"]
DISTRIBUTIONS = {
    "iid": ("homogeneous_partition", ""),
    "label_quantity_C1": ("label_distribution\\label_quantity\\C1", "labels_per_client1"),
    "label_quantity_C2": ("label_distribution\\label_quantity\\C2", "labels_per_client2"),
    "label_quantity_C3": ("label_distribution\\label_quantity\\C3", "labels_per_client3"),
    "dirichlet": ("label_distribution\\dirichlet\\alpha0.5", "alpha0.5"),
    "iid_noniid": ("feature_distribution", ""),
    "noise": ("quantity_skew", "")
}
METRICS = ["Test_Accuracy", "Centralized_Loss", "Distributed_Loss", "Distributed_Evaluate_Accuracy"]
ROUNDS = 10  # 11 points (0 à 10)

# Couleurs pour chaque distribution
COLORS = {
    "iid": "#1f77b4",
    "label_quantity_C1": "#ff7f0e",
    "label_quantity_C2": "#2ca02c",
    "label_quantity_C3": "#d62728",
    "dirichlet": "#9467bd",
    "iid_noniid": "#8c564b",
    "noise": "#e377c2"
}

# Marqueurs différents pour chaque distribution
MARKERS = {
    "iid": "o",          # Cercle
    "label_quantity_C1": "^",  # Triangle vers le haut
    "label_quantity_C2": "v",  # Triangle vers le bas
    "label_quantity_C3": "s",  # Carré
    "dirichlet": "D",    # Diamant
    "iid_noniid": "p",   # Pentagone
    "noise": "*"         # Étoile
}

def read_csv_file(filepath):
    """Lit un fichier CSV et retourne les données sous forme de dictionnaire."""
    data = {metric: [] for metric in METRICS}
    rounds = []
    
    # Vérifier si le fichier existe
    if not os.path.exists(filepath):
        print(f"Fichier non trouvé : {filepath}")
        print(f"Chemin absolu : {os.path.abspath(filepath)}")
        print(f"Répertoire de travail actuel : {os.getcwd()}")
        return None, None
    
    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rounds.append(int(row["Round"]))
                for metric in METRICS:
                    data[metric].append(float(row[metric]))
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {filepath} : {e}")
        return None, None
    return rounds, data

def plot_metric_for_dataset(dataset, metric, all_data, epochs):
    """Crée un plot pour une métrique donnée, un dataset donné et un nombre d'époques donné."""
    plt.figure(figsize=(10, 6))
    
    # Collecter toutes les valeurs pour ajuster les limites
    all_values = []
    for dist_name, data in all_data[dataset].items():
        rounds = data["rounds"]
        values = data["metrics"][metric]
        # Utiliser un marqueur différent pour chaque distribution
        plt.plot(rounds, values, label=dist_name, color=COLORS[dist_name], linewidth=2, 
                 marker=MARKERS[dist_name], markersize=5)
        all_values.extend(values)
    
    plt.xlabel("Round", fontsize=12)
    plt.ylabel(metric.replace("_", " "), fontsize=12)
    plt.title(f"{metric.replace('_', ' ')} for {dataset.upper()} Dataset (Epochs={epochs})", fontsize=14, pad=15)
    plt.legend(loc="best", fontsize=10)
    
    # Supprimer la grille pour un rendu plus propre
    plt.grid(False)
    # Alternative : grille très discrète
    # plt.grid(True, linestyle=':', alpha=0.3, which='both')
    
    plt.xticks(range(ROUNDS + 1), fontsize=10)
    plt.yticks(fontsize=10)
    
    # Ajuster les limites de l'axe des ordonnées dynamiquement
    if metric in ["Test_Accuracy", "Distributed_Evaluate_Accuracy"]:
        plt.ylim(0, 1)  # Précision entre 0 et 1
    else:
        # Calculer le maximum des valeurs avec une marge de 10%
        if all_values:  # Vérifier que all_values n'est pas vide
            max_value = max(all_values)
            margin = max_value * 0.1  # Marge de 10%
            plt.ylim(bottom=0, top=max_value + margin)
        else:
            plt.ylim(bottom=0, top=1)  # Valeur par défaut si pas de données
    
    plt.tight_layout()
    
    # Sauvegarder le plot dans un sous-dossier par dataset
    output_dir = os.path.join("results", "plots", f"epochs{epochs}", dataset)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{metric}_{dataset}_epochs{epochs}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main(dataset_filter=None, epochs_filter=None):
    """Génère les plots, avec des filtres optionnels pour dataset et époques."""
    print(f"Répertoire de travail actuel : {os.getcwd()}")
    print(f"Chemin absolu de BASE_PATH : {BASE_PATH}")
    
    # Définir les époques à traiter
    epochs_to_process = [epochs_filter] if epochs_filter is not None else [1, 2]
    
    # Boucle sur les époques
    for epochs in epochs_to_process:
        print(f"\nGénération des plots pour epochs={epochs}...")
        
        # Structure pour stocker toutes les données : {dataset: {distribution: {"rounds": [], "metrics": {metric: []}}}}
        all_data = {dataset: {} for dataset in DATASETS}
        
        # Filtrer les datasets à traiter
        datasets_to_process = [dataset_filter] if dataset_filter else DATASETS
        
        # Lire tous les fichiers CSV pour l'époque donnée
        for dataset in datasets_to_process:
            for dist_name, (dist_path, param) in DISTRIBUTIONS.items():
                # Construire le chemin du fichier CSV avec un double underscore
                param_str = f"_{param}" if param else ""
                filepath = os.path.join(
                    BASE_PATH, dataset, dist_path, "metrics",
                    f"rounds{ROUNDS}_epochs{epochs}__run1.csv" if not param else f"rounds{ROUNDS}_epochs{epochs}_{param}_run1.csv"
                )
                rounds, data = read_csv_file(filepath)
                if rounds is None or data is None:
                    print(f"Skipping {dataset}/{dist_name} (fichier {filepath} non trouvé)")
                    continue
                all_data[dataset][dist_name] = {
                    "rounds": rounds,
                    "metrics": data
                }
        
        # Générer les plots pour les datasets et l'époque sélectionnés
        for metric in METRICS:
            for dataset in datasets_to_process:
                if not all_data[dataset]:  # Si aucune donnée pour ce dataset
                    print(f"Aucune donnée pour {dataset}, skipping.")
                    continue
                plot_metric_for_dataset(dataset, metric, all_data, epochs)

if __name__ == "__main__":
    # Ajouter des arguments pour spécifier un dataset et une époque
    parser = argparse.ArgumentParser(description="Générer des plots pour un ou tous les datasets et époques.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASETS,
        default=None,
        help="Dataset à visualiser (par exemple, 'mnist'). Si non spécifié, tous les datasets seront traités."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        choices=[1, 2],
        default=None,
        help="Nombre d'époques à visualiser (1 ou 2). Si non spécifié, les deux seront traités."
    )
    args = parser.parse_args()
    
    main(dataset_filter=args.dataset, epochs_filter=args.epochs)