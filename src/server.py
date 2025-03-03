import os
import pickle
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from src.utils import get_weights
from models.lstm import LSTM


# Define function to save model weights
def save_model_weights(parameters, round_num, save_dir="models/saved_models", save_every=5):
    if round_num % save_every == 0:  # Save every 'save_every' rounds
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"model_round_{round_num}.pkl")

        # Convert parameters to ndarrays
        ndarrays = parameters_to_ndarrays(parameters)

        # Save as pickle
        with open(filepath, "wb") as f:
            pickle.dump(ndarrays, f)

        print(f"Model saved at round {round_num} to {filepath}")


round_num = 0


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(LSTM())
    parameters = ndarrays_to_parameters(ndarrays)

    def aggregate_fit_metrics(metrics):
        # Save model every 'save_every' rounds
        global round_num
        round_num += 1
        save_model_weights(parameters, round_num)

        return {"avg_train_loss": sum(d[1]["train_loss"] for d in metrics) / len(metrics)}

    def aggregate_evaluate_metrics(metrics):
        avg_accuracy = sum(d[1]["accuracy"] for d in metrics) / len(metrics)
        avg_precision = sum(d[1]["precision"] for d in metrics) / len(metrics)
        avg_recall = sum(d[1]["recall"] for d in metrics) / len(metrics)
        avg_f1 = sum(d[1]["f1_score"] for d in metrics) / len(metrics)

        return {
            "avg_accuracy": avg_accuracy,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1_score": avg_f1,
        }

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
