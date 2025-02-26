import torch
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
        train_loss = train(self.model, self.train_loader, self.local_epochs, self.device)
        return get_weights(self.model), len(self.train_loader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader, self.device)
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


client_counter = 0
client_id_map = {}


def client_fn(context: Context):
    global client_counter
    node_id = context.node_id
    client_id = client_counter + 1
    client_counter += 1
    client_id_map[node_id] = client_id

    print(f"Client: {node_id}, with id: {client_id}, is connecting...")
    # Load model and data
    net = LSTM()
    train_loader, test_loader = load_data(client_id)
    print(f"Loaded data for client id: {client_id}")
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, train_loader, test_loader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
