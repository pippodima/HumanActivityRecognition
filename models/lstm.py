import torch.nn as nn


# class LSTM(nn.Module):
#     def __init__(self, n_features=18, hidden_dim=64, n_outputs=15):
#         super(LSTM, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.n_features = n_features
#         self.n_outputs = n_outputs
#         self.hidden = None
#         self.cell = None
#         # Simple LSTM
#         self.basic_rnn = nn.LSTM(self.n_features, self.hidden_dim, batch_first=True)
#         # Classifier to produce as many logits as outputs
#         self.classifier = nn.Linear(self.hidden_dim, self.n_outputs)
#
#     def forward(self, X):
#         # X is batch first (N, L, F)
#         # output is (N, L, H)
#         # final hidden state is (1, N, H)
#         # final cell state is (1, N, H)
#         batch_first_output, (self.hidden, self.cell) = self.basic_rnn(X)
#
#         # only last item in sequence (N, 1, H)
#         last_output = batch_first_output[:, -1]
#         # classifier will output (N, 1, n_outputs)
#         out = self.classifier(last_output)
#
#         # final output is (N, n_outputs)
#         return out.view(-1, self.n_outputs)



class LSTM(nn.Module):
    def __init__(self, n_features=18, hidden_dim=64, n_outputs=15, num_layers=2, dropout_prob=0.2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.num_layers = num_layers

        # Multi-layer LSTM with dropout
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob  # Dropout between LSTM layers
        )

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        # Fully connected layer with weight regularization
        self.classifier = nn.Linear(hidden_dim, n_outputs)

        # Dropout before final classification
        self.final_dropout = nn.Dropout(dropout_prob)

    def forward(self, X):
        # LSTM forward pass
        lstm_out, _ = self.lstm(X)

        # Extract last time step output
        last_output = lstm_out[:, -1, :]

        # Apply batch normalization
        normalized_output = self.batch_norm(last_output)

        # Apply final dropout
        out = self.final_dropout(normalized_output)

        # Fully connected layer
        out = self.classifier(out)

        return out
