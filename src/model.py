
import torch
import torch.nn as nn

class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()

        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h

        dec_layers = []
        reversed_dims = list(reversed(hidden_dims[:-1]))
        prev = hidden_dims[-1]
        for h in reversed_dims:
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
