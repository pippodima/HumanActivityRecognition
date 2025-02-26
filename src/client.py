import torch
import threading
from logger import script_logger
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from src.utils import get_weights, load_data, set_weights, test, train
from models.lstm import LSTM


class FlowerClient(NumPyClient):
    def __init__(self, model, train_loader, test_loader, local_epochs):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        script_logger.info("Training Started")
        train_loss = train(self.model, self.train_loader, self.local_epochs, self.device)
        script_logger.info("Training Ended")
        return get_weights(self.model), len(self.train_loader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader, self.device)
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}




def client_fn(context: Context):
    client_id = context.node_config['partition-id'] + 1
    script_logger.info(f"Loading model and data for client id: {client_id}")
    net = LSTM()
    train_loader, test_loader = load_data(client_id)
    script_logger.info(f"Data loaded successfully for client id: {client_id}")
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, train_loader, test_loader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
