"""
Variational Autoencoder architecture with LeakyReLU activations.
Encoder produces μ and logσ²; Decoder reconstructs mean+logvar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder network: input → hidden1 → hidden2 → (μ, logvar).
    """
    def __init__(self, input_dim: int, h1: int, h2: int, z_dim: int):
        super().__init__()
        self.fc1       = nn.Linear(input_dim, h1)
        self.fc2       = nn.Linear(h1, h2)
        self.fc_mu     = nn.Linear(h2, z_dim)
        self.fc_logvar = nn.Linear(h2, z_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch, input_dim] tensor

        Returns:
            mu, logvar: each [batch, z_dim]
        """
        h = F.leaky_relu(self.fc1(x), 0.01)
        h = F.leaky_relu(self.fc2(h), 0.01)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    """
    Decoder network: z → hidden1 → hidden2 → (recon_mu, recon_logvar).
    """
    def __init__(self, z_dim: int, h1: int, h2: int, out_dim: int):
        super().__init__()
        self.fc1       = nn.Linear(z_dim, h1)
        self.fc2       = nn.Linear(h1, h2)
        self.fc_out    = nn.Linear(h2, out_dim)
        self.fc_logvar = nn.Linear(h2, out_dim)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [batch, z_dim]

        Returns:
            recon_mu, recon_logvar: each [batch, out_dim]
        """
        h = F.leaky_relu(self.fc1(z), 0.01)
        h = F.leaky_relu(self.fc2(h), 0.01)
        return self.fc_out(h), self.fc_logvar(h)