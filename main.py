import os
import csv
import argparse
from omegaconf import OmegaConf
import torch
from fedml.algorithms.scaffold.simulation import run_scaffold


def main():
    parser = argparse.ArgumentParser(description="Run Federated Averaging Simulations")
    parser.add_argument("--method", type=str, default="scaffold",
                        help="Federated learning method to use")
    parser.add_argument("--partitioning", type=str, required=True,
                        choices=["iid", "label_quantity", "dirichlet", "iid_noniid", "noise"],
                        help="Partitioning strategy")
    parser.add_argument("--dataset", type=str, default="mnist",
                        help="Dataset to use")
    parser.add_argument("--num_clients", type=int, default=10,
                        help="Number of clients")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of federation rounds")
    parser.add_argument("--epochs", type=int,nargs="+",
                        help="Number of local epochs")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of independent runs")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate for the optimizer")
    
    # Partition-specific parameters
    parser.add_argument("--labels_per_client", type=int, nargs="+",
                        help="Number of labels per client (for label_quantity)")
    parser.add_argument("--alpha", type=float, nargs="+",
                        help="Dirichlet concentration parameter")
    parser.add_argument("--similarity", type=float,
                        help="IID-ness percentage for iid_noniid")
    parser.add_argument("--segma", type=float,
                        help="Noise level for noise partitioning")
    
    # Model/backend config
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--num_cpus", type=int, default=6,
                        help="Number of CPUs per client")
    parser.add_argument("--num_gpus", type=float, default=0,
                        help="Number of GPUs per client")
    
    args = parser.parse_args()
    if args.method not in ["scaffold"]:  # Add more methods as needed
        raise ValueError("Invalid method")

    # Validate partition-specific parameters
    if args.partitioning == "label_quantity" and not args.labels_per_client:
        raise ValueError("--labels_per_client required for label_quantity partitioning")
    if args.partitioning == "dirichlet" and not args.alpha:
        raise ValueError("--alpha required for dirichlet partitioning")
    if args.partitioning == "iid_noniid" and not args.similarity:
        raise ValueError("--similarity required for iid_noniid partitioning")
    if args.partitioning == "noise" and not args.segma:
        raise ValueError("--segma required for noise partitioning")
    # epochs validation
    
    # Model configuration
    if args.dataset == "mnist" or args.dataset == "fmnist":
        model_cfg = OmegaConf.create({
            "_target_": "fedml.models.MNISTModel" ,
            "input_dim": 256,
            "hidden_dims": [120, 84],
            "num_classes": 10,
        })
    elif args.dataset == "cifar10":
        model_cfg = OmegaConf.create({
            "_target_": "fedml.models.CNN",
            "input_dim": 400,
            "hidden_dims": [120, 84],
            "num_classes": 10,
        })

    # Backend configuration
    backend_config = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }

    # Parameter combinations to process
    param_combinations = [{}]
    if args.partitioning == "label_quantity":
        param_combinations = [{"labels_per_client": c} for c in args.labels_per_client]
    elif args.partitioning == "dirichlet":
        param_combinations = [{"alpha": a} for a in args.alpha]

    if args.epochs is None:
        raise ValueError("--epochs required")
    else:
        epochs = [int(e) for e in args.epochs]

    for params in param_combinations:
        # Create data configuration
        data_config = {
            "name": args.dataset,
            "partitioning": args.partitioning,
            "batch_size": args.batch_size,
        }

        # Add partition-specific parameters
        if args.partitioning == "label_quantity":
            data_config["labels_per_client"] = params["labels_per_client"]
        elif args.partitioning == "dirichlet":
            data_config["alpha"] = params.get("alpha", 0.5)
        elif args.partitioning == "iid_noniid":
            data_config["similarity"] = args.similarity
        elif args.partitioning == "noise":
            data_config["segma"] = args.segma

        data_config = OmegaConf.create(data_config)

        # Create results directory
        results_dir = os.path.join(
            "results",
            "experiments",
            args.method,
            args.dataset,
            _get_partition_subdir(args.partitioning, params),
            "metrics"
        )
        os.makedirs(results_dir, exist_ok=True)

        # Run multiple independent simulations
        for epoch in epochs:
            for run in range(1, args.runs + 1):
                print(f"\n▶︎ {args.partitioning.upper()} Simulation")
                print(f" - Parameters: {params}")
                print(f" - Num epochs: {epoch}")
                print(f" - Run {run}/{args.runs}")
                model_dir=f"results/weights/{args.method}/{args.dataset}/{args.partitioning}/epoch{epoch}"
                if args.partitioning == "label_quantity":
                    model_dir += f"/C{params['labels_per_client']}"
                elif args.partitioning == "dirichlet":
                    model_dir += f"/alpha{params['alpha']}"
                print(model_dir)
                model_dir += f"/run{run}"
                
                history = run_scaffold(
                        data_config=data_config,
                        model_cfg=model_cfg,
                        backend_config=backend_config,
                        num_clients=args.num_clients,
                        num_rounds=args.rounds,
                        num_epochs=epoch,
                        learning_rate= args.learning_rate,
                        model_dir=model_dir,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                
                # Process and save results
                _save_results(
                    history=history,
                    results_dir=results_dir,
                    params=params,
                    rounds=args.rounds,
                    epochs=epoch,
                    run=run
                )

    print("\nAll simulations completed!")

def _get_partition_subdir(partitioning, params):
    """Generate appropriate subdirectory structure based on partitioning."""
    if partitioning == "iid":
        return "homogeneous_partition"
    elif partitioning == "label_quantity":
        return f"label_distribution/label_quantity/C{params['labels_per_client']}"
    elif partitioning == "dirichlet":
        return f"label_distribution/dirichlet/alpha{params.get('alpha', 0.5)}"
    elif partitioning == "iid_noniid":
        return "quantity_skew"
    elif partitioning == "noise":
        return "feature_distribution"
    elif partitioning == "synthetic":
        return "synthetic_partition"
    elif partitioning == "real-world":
        return "real_world_partition"
    return "other"

def _save_results(history, results_dir, params, rounds, epochs, run):
    """Save simulation results to CSV file."""
    centralized_metrics = history.metrics_centralized
    accuracies = [round(value, 4) for _, value in centralized_metrics["test_accuracy"]]
    rounds_list = list(range(0, rounds + 1))

    # Generate filename components
    param_str = "_".join([f"{k}{v}" for k, v in params.items()])
    filename = f"rounds{rounds}_epochs{epochs}_{param_str}_run{run}.csv"

    # Write to CSV
    with open(os.path.join(results_dir, filename), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Accuracy"])
        writer.writerows(zip(rounds_list, accuracies))

    print(f"Results saved to {os.path.join(results_dir, filename)}")

if __name__ == "__main__":
    main()
