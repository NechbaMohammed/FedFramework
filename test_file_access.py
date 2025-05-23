import os

files_to_check = [
    "results/experiments/tamuna\\mnist\\homogeneous_partition\\metrics\\rounds10_epochs1__run1.csv",
    "results\\experiments\\tamuna\\mnist\\feature_distribution\\metrics\\rounds10_epochs1__run1.csv",
    "results\\experiments\\tamuna\\mnist\\quantity_skew\\metrics\\rounds10_epochs1__run1.csv"
]

for filepath in files_to_check:
    if os.path.exists(filepath):
        print(f"Fichier trouvé : {filepath}")
        try:
            with open(filepath, "r") as f:
                print(f"Fichier accessible : {filepath}")
        except Exception as e:
            print(f"Erreur d'accès au fichier {filepath} : {e}")
    else:
        print(f"Fichier non trouvé : {filepath}")