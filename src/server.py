from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from src.utils import get_weights
from models.lstm import LSTM


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(LSTM())
    parameters = ndarrays_to_parameters(ndarrays)

    def aggregate_fit_metrics(metrics):
        return {"avg_train_loss": sum(d[1]["train_loss"] for d in metrics) / len(metrics)}

    def aggregate_evaluate_metrics(metrics):
        return {"avg_accuracy": sum(d[1]["accuracy"] for d in metrics) / len(metrics)}

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
