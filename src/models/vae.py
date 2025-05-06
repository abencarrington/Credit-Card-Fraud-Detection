"""
Variational Autoencoder implementation with:
1. KL annealing for stable training
2. β-VAE support for better disentanglement
3. Improved model architecture with batch normalization
4. Save/load functionality for model persistence
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder network with batch normalization and dropout.
    
    Maps input data to latent space parameters (μ, logσ²).
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden1: int, 
        hidden2: int, 
        z_dim: int,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc_mu = nn.Linear(hidden2, z_dim)
        self.fc_logvar = nn.Linear(hidden2, z_dim)
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass through encoder network.
        
        Args:
            x: [batch, input_dim] tensor of input data
            
        Returns:
            mu: [batch, z_dim] tensor of means
            logvar: [batch, z_dim] tensor of log variances
        """
        h = F.leaky_relu(self.bn1(self.fc1(x)), 0.01)
        h = self.dropout1(h)
        h = F.leaky_relu(self.bn2(self.fc2(h)), 0.01)
        h = self.dropout2(h)
        
        return self.fc_mu(h), self.fc_logvar(h)
    
    def save(self, path: str):
        """Save encoder state dict to file."""
        torch.save(self.state_dict(), path)
        
    @classmethod
    def load(cls, path: str, **kwargs):
        """Load encoder from saved state dict."""
        instance = cls(**kwargs)
        instance.load_state_dict(torch.load(path))
        return instance


class Decoder(nn.Module):
    """
    Decoder network with batch normalization.
    
    Maps latent variables back to observation space parameters.
    """
    def __init__(
        self, 
        z_dim: int, 
        hidden1: int, 
        hidden2: int, 
        out_dim: int,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc_out = nn.Linear(hidden2, out_dim)
        self.fc_logvar = nn.Linear(hidden2, out_dim)
        
    def forward(self, z: torch.Tensor):
        """
        Forward pass through decoder network.
        
        Args:
            z: [batch, z_dim] tensor of latent variables
            
        Returns:
            recon_mu: [batch, out_dim] tensor of reconstructed means
            recon_logvar: [batch, out_dim] tensor of reconstructed log variances
        """
        h = F.leaky_relu(self.bn1(self.fc1(z)), 0.01)
        h = self.dropout1(h)
        h = F.leaky_relu(self.bn2(self.fc2(h)), 0.01)
        h = self.dropout2(h)
        
        return self.fc_out(h), self.fc_logvar(h)
    
    def save(self, path: str):
        """Save decoder state dict to file."""
        torch.save(self.state_dict(), path)
        
    @classmethod
    def load(cls, path: str, **kwargs):
        """Load decoder from saved state dict."""
        instance = cls(**kwargs)
        instance.load_state_dict(torch.load(path))
        return instance


class GlobalVAE(nn.Module):
    """
    Combined VAE model with encoder and decoder components.
    
    Includes KL annealing and β-VAE functionality.
    """
    def __init__(
        self,
        input_dim: int,
        hidden1: int = 128,
        hidden2: int = 64,
        zdim: int = 2,
        beta: float = 1.0,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden1, hidden2, zdim, dropout_rate)
        self.decoder = Decoder(zdim, hidden2, hidden1, input_dim, dropout_rate)
        self.zdim = zdim
        self.beta = beta
        
    def encode(self, x: torch.Tensor):
        """Encode input to latent parameters."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor):
        """Decode latent variables to reconstructed parameters."""
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor):
        """
        Full forward pass through VAE.
        
        Args:
            x: [batch, input_dim] tensor of input data
            
        Returns:
            recon_x: [batch, input_dim] tensor of reconstructed means
            recon_logvar: [batch, input_dim] tensor of reconstructed log variances
            mu: [batch, zdim] tensor of encoded means
            logvar: [batch, zdim] tensor of encoded log variances
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_mu, recon_logvar = self.decode(z)
        
        return recon_mu, recon_logvar, mu, logvar
    
    def compute_loss(
        self,
        x: torch.Tensor,
        annealing_factor: float = 1.0,
        reconstruction_weight: float = 1.0
    ):
        """
        Compute ELBO loss with KL annealing and β-weighting.
        
        Args:
            x: [batch, input_dim] tensor of input data
            annealing_factor: Factor for KL annealing (0-1)
            reconstruction_weight: Weight for reconstruction term
            
        Returns:
            total_loss: Combined ELBO loss value
            recon_loss: Reconstruction loss component
            kl_loss: KL divergence loss component
        """
        # Forward pass
        recon_mu, recon_logvar, mu, logvar = self.forward(x)
        
        # Reconstruction loss: negative log likelihood
        recon_var = torch.exp(recon_logvar)
        recon_loss = 0.5 * (
            torch.log(2 * torch.pi * recon_var) +
            (x - recon_mu).pow(2) / recon_var
        ).sum(dim=1).mean()
        
        # KL divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), 
            dim=1
        ).mean()
        
        # Combined loss with annealing and beta weighting
        total_loss = (
            reconstruction_weight * recon_loss + 
            annealing_factor * self.beta * kl_loss
        )
        
        return total_loss, recon_loss, kl_loss
    
    def reconstruct(self, x: torch.Tensor):
        """
        Reconstruct input data through the VAE.
        
        Args:
            x: [batch, input_dim] tensor of input data
            
        Returns:
            recon_x: [batch, input_dim] tensor of reconstructed data
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_mu, _ = self.decode(z)
        return recon_mu
    
    def generate(self, n_samples: int, device: torch.device):
        """
        Generate new samples from the latent space.
        
        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            samples: [n_samples, input_dim] tensor of generated samples
        """
        z = torch.randn(n_samples, self.zdim, device=device)
        recon_mu, _ = self.decode(z)
        return recon_mu
    
    def compute_reconstruction_error(self, x: torch.Tensor):
        """
        Compute reconstruction error for anomaly detection.
        
        Args:
            x: [batch, input_dim] tensor of input data
            
        Returns:
            errors: [batch] tensor of reconstruction errors
        """
        recon_mu, recon_logvar, _, _ = self.forward(x)
        recon_var = torch.exp(recon_logvar)
        
        # Negative log likelihood as reconstruction error
        errors = 0.5 * (
            torch.log(2 * torch.pi * recon_var) +
            (x - recon_mu).pow(2) / recon_var
        ).sum(dim=1)
        
        return errors
    
    def save(self, path: str):
        """
        Save complete VAE model (encoder and decoder).
        
        Args:
            path: Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'beta': self.beta,
            'zdim': self.zdim
        }, path)
        
    @classmethod
    def load(cls, path: str, input_dim: int, **kwargs):
        """
        Load complete VAE model from saved state dict.
        
        Args:
            path: Path to load model from
            input_dim: Input dimension
            **kwargs: Additional parameters for model initialization
            
        Returns:
            model: Loaded VAE model
        """
        checkpoint = torch.load(path)
        
        model = cls(
            input_dim=input_dim,
            zdim=checkpoint.get('zdim', 2),
            beta=checkpoint.get('beta', 1.0),
            **kwargs
        )
        
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
        
        return model


class CustomerVAE(GlobalVAE):
    """
    Customer-specific VAE model extending the GlobalVAE.
    
    Includes additional customer-specific functionality.
    """
    def __init__(
        self,
        input_dim: int,
        customer_id: int,
        hidden1: int = 64,
        hidden2: int = 32,
        zdim: int = 2,
        beta: float = 1.0,
        dropout_rate: float = 0.2
    ):
        super().__init__(
            input_dim=input_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            zdim=zdim,
            beta=beta,
            dropout_rate=dropout_rate
        )
        self.customer_id = customer_id
        
    def save(self, path: str):
        """
        Save customer-specific VAE model.
        
        Args:
            path: Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'beta': self.beta,
            'zdim': self.zdim,
            'customer_id': self.customer_id
        }, path)
        
    @classmethod
    def load(cls, path: str, input_dim: int, **kwargs):
        """
        Load customer-specific VAE model.
        
        Args:
            path: Path to load model from
            input_dim: Input dimension
            **kwargs: Additional parameters for model initialization
            
        Returns:
            model: Loaded customer VAE model
        """
        checkpoint = torch.load(path)
        
        model = cls(
            input_dim=input_dim,
            customer_id=checkpoint.get('customer_id', 0),
            zdim=checkpoint.get('zdim', 2),
            beta=checkpoint.get('beta', 1.0),
            **kwargs
        )
        
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
        
        return model