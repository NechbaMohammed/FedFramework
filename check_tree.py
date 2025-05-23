import os

def print_tree(directory, indent=""):
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isfile(path):
            print(f"{indent}├── {item}")
        elif os.path.isdir(path):
            print(f"{indent}└── {item}\\")
            print_tree(path, indent + "    ")

if __name__ == "__main__":
    print_tree("results")