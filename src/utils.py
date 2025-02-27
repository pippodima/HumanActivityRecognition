from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from logger import script_logger


def create_lstm_windows(user_number, seconds=1, step=1, test_size=0.2, random_state=42):
    script_logger.info(f"Reading data for user {user_number}")
    df = pd.read_csv(f"data/labeled_sensor_data_merged_user-{user_number}.csv")
    df = df.drop(columns="ts")

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])

    features = df.drop(columns=["label"]).values
    labels = df["label"].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    window_size = 200 * seconds  # 200 measurements = 1s
    stride = 200 * step
    X, y = [], []

    for i in range(0, len(features) - window_size + 1, stride):  # Step by 200 (1 second)
        chunk_labels = labels[i: i + window_size]

        # Check if all labels in the 1-second window are the same
        if np.all(chunk_labels == chunk_labels[0]):
            X.append(features[i: i + window_size])  # Store the 200-sample window
            y.append(chunk_labels[0])  # Store the single label

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


    return X_train, X_test, y_train, y_test


def load_data(client_id):
    X_train, X_test, y_train, y_test = create_lstm_windows(user_number=client_id, seconds=3, step=1)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    return train_loader, test_loader


def train(model, train_loader, num_epochs, device):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(num_epochs):
        total_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)



def test(model, test_loader, device):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    correct, total = 0, 0
    loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()
    accuracy = correct / total
    loss = loss / len(test_loader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
