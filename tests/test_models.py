"""
Unit tests for the VAE models implementation.

Tests cover:
- Model initialization with correct dimensions
- Forward pass functionality
- Save/load functionality
- KL annealing and beta-weighting
- Loss function behavior
"""

import os
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from src.models.vae import Encoder, Decoder, GlobalVAE, CustomerVAE
from src.bootstrap import set_project_root

# Set project root to resolve imports correctly
set_project_root()

class TestVAEModels(unittest.TestCase):
    """Test cases for VAE model implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set fixed random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Set test dimensions
        self.input_dim = 10
        self.hidden1 = 8
        self.hidden2 = 6
        self.zdim = 2
        self.batch_size = 5
        
        # Create test data
        self.test_input = torch.randn(self.batch_size, self.input_dim)
        
        # Use CPU device for testing
        self.device = torch.device("cpu")
    
    def test_encoder_initialization(self):
        """Test encoder model initialization."""
        encoder = Encoder(
            input_dim=self.input_dim,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            z_dim=self.zdim
        )
        
        # Check model structure
        self.assertEqual(encoder.fc1.in_features, self.input_dim)
        self.assertEqual(encoder.fc1.out_features, self.hidden1)
        self.assertEqual(encoder.fc2.in_features, self.hidden1)
        self.assertEqual(encoder.fc2.out_features, self.hidden2)
        self.assertEqual(encoder.fc_mu.in_features, self.hidden2)
        self.assertEqual(encoder.fc_mu.out_features, self.zdim)
        self.assertEqual(encoder.fc_logvar.in_features, self.hidden2)
        self.assertEqual(encoder.fc_logvar.out_features, self.zdim)
    
    def test_decoder_initialization(self):
        """Test decoder model initialization."""
        decoder = Decoder(
            z_dim=self.zdim,
            hidden1=self.hidden2,
            hidden2=self.hidden1,
            out_dim=self.input_dim
        )
        
        # Check model structure
        self.assertEqual(decoder.fc1.in_features, self.zdim)
        self.assertEqual(decoder.fc1.out_features, self.hidden2)
        self.assertEqual(decoder.fc2.in_features, self.hidden2)
        self.assertEqual(decoder.fc2.out_features, self.hidden1)
        self.assertEqual(decoder.fc_out.in_features, self.hidden1)
        self.assertEqual(decoder.fc_out.out_features, self.input_dim)
        self.assertEqual(decoder.fc_logvar.in_features, self.hidden1)
        self.assertEqual(decoder.fc_logvar.out_features, self.input_dim)
    
    def test_encoder_forward(self):
        """Test encoder forward pass."""
        encoder = Encoder(
            input_dim=self.input_dim,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            z_dim=self.zdim
        )
        
        # Forward pass
        mu, logvar = encoder(self.test_input)
        
        # Check output shapes
        self.assertEqual(mu.shape, (self.batch_size, self.zdim))
        self.assertEqual(logvar.shape, (self.batch_size, self.zdim))
    
    def test_decoder_forward(self):
        """Test decoder forward pass."""
        decoder = Decoder(
            z_dim=self.zdim,
            hidden1=self.hidden2,
            hidden2=self.hidden1,
            out_dim=self.input_dim
        )
        
        # Create latent vector
        z = torch.randn(self.batch_size, self.zdim)
        
        # Forward pass
        recon_mu, recon_logvar = decoder(z)
        
        # Check output shapes
        self.assertEqual(recon_mu.shape, (self.batch_size, self.input_dim))
        self.assertEqual(recon_logvar.shape, (self.batch_size, self.input_dim))
    
    def test_global_vae_initialization(self):
        """Test global VAE model initialization."""
        vae = GlobalVAE(
            input_dim=self.input_dim,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            zdim=self.zdim,
            beta=2.0
        )
        
        # Check model structure
        self.assertIsInstance(vae.encoder, Encoder)
        self.assertIsInstance(vae.decoder, Decoder)
        self.assertEqual(vae.zdim, self.zdim)
        self.assertEqual(vae.beta, 2.0)
    
    def test_global_vae_forward(self):
        """Test global VAE forward pass."""
        vae = GlobalVAE(
            input_dim=self.input_dim,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            zdim=self.zdim
        )
        
        # Forward pass
        recon_mu, recon_logvar, mu, logvar = vae(self.test_input)
        
        # Check output shapes
        self.assertEqual(recon_mu.shape, (self.batch_size, self.input_dim))
        self.assertEqual(recon_logvar.shape, (self.batch_size, self.input_dim))
        self.assertEqual(mu.shape, (self.batch_size, self.zdim))
        self.assertEqual(logvar.shape, (self.batch_size, self.zdim))
    
    def test_global_vae_loss(self):
        """Test global VAE loss calculation."""
        vae = GlobalVAE(
            input_dim=self.input_dim,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            zdim=self.zdim,
            beta=1.0
        )
        
        # Calculate loss with default parameters
        total_loss, recon_loss, kl_loss = vae.compute_loss(self.test_input)
        
        # Check loss values
        self.assertTrue(isinstance(total_loss, torch.Tensor))
        self.assertTrue(isinstance(recon_loss, torch.Tensor))
        self.assertTrue(isinstance(kl_loss, torch.Tensor))
        self.assertTrue(total_loss.item() > 0)
        self.assertTrue(recon_loss.item() > 0)
        self.assertTrue(kl_loss.item() > 0)
        
        # KL annealing test
        total_loss_annealed, _, kl_loss_annealed = vae.compute_loss(
            self.test_input, annealing_factor=0.5
        )
        
        # Check that KL contribution is reduced with annealing
        self.assertLess(total_loss_annealed.item(), total_loss.item())
        
        # Beta weighting test
        vae.beta = 2.0
        total_loss_beta, _, kl_loss_beta = vae.compute_loss(self.test_input)
        
        # Check that KL contribution is increased with higher beta
        self.assertGreater(total_loss_beta.item(), total_loss.item())
    
    def test_customer_vae_initialization(self):
        """Test customer VAE model initialization."""
        customer_id = 12345
        vae = CustomerVAE(
            input_dim=self.input_dim,
            customer_id=customer_id,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            zdim=self.zdim,
            beta=1.5
        )
        
        # Check model structure
        self.assertIsInstance(vae.encoder, Encoder)
        self.assertIsInstance(vae.decoder, Decoder)
        self.assertEqual(vae.zdim, self.zdim)
        self.assertEqual(vae.beta, 1.5)
        self.assertEqual(vae.customer_id, customer_id)
    
    def test_vae_save_load(self):
        """Test VAE model save and load functionality."""
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "test_model.pt")
            
            # Create and save model
            original_model = GlobalVAE(
                input_dim=self.input_dim,
                hidden1=self.hidden1,
                hidden2=self.hidden2,
                zdim=self.zdim,
                beta=1.5
            )
            
            # Run a forward pass to initialize lazy layers if any
            _ = original_model(self.test_input)
            
            # Get original weights for comparison
            original_encoder_weight = original_model.encoder.fc1.weight.data.clone()
            original_decoder_weight = original_model.decoder.fc1.weight.data.clone()
            
            # Save model
            original_model.save(model_path)
            
            # Load model
            loaded_model = GlobalVAE.load(model_path, input_dim=self.input_dim)
            
            # Compare model parameters
            loaded_encoder_weight = loaded_model.encoder.fc1.weight.data
            loaded_decoder_weight = loaded_model.decoder.fc1.weight.data
            
            # Check model metadata
            self.assertEqual(loaded_model.zdim, self.zdim)
            self.assertEqual(loaded_model.beta, 1.5)
            
            # Check model weights
            self.assertTrue(torch.allclose(original_encoder_weight, loaded_encoder_weight))
            self.assertTrue(torch.allclose(original_decoder_weight, loaded_decoder_weight))
    
    def test_customer_vae_save_load(self):
        """Test customer VAE model save and load functionality."""
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "test_customer_model.pt")
            customer_id = 54321
            
            # Create and save model
            original_model = CustomerVAE(
                input_dim=self.input_dim,
                customer_id=customer_id,
                hidden1=self.hidden1,
                hidden2=self.hidden2,
                zdim=self.zdim,
                beta=1.8
            )
            
            # Run a forward pass to initialize lazy layers if any
            _ = original_model(self.test_input)
            
            # Save model
            original_model.save(model_path)
            
            # Load model
            loaded_model = CustomerVAE.load(model_path, input_dim=self.input_dim)
            
            # Check model metadata
            self.assertEqual(loaded_model.zdim, self.zdim)
            self.assertEqual(loaded_model.beta, 1.8)
            self.assertEqual(loaded_model.customer_id, customer_id)
    
    def test_reconstruction_error(self):
        """Test reconstruction error calculation."""
        vae = GlobalVAE(
            input_dim=self.input_dim,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            zdim=self.zdim
        )
        
        # Compute reconstruction error
        errors = vae.compute_reconstruction_error(self.test_input)
        
        # Check output shape and values
        self.assertEqual(errors.shape, (self.batch_size,))
        self.assertTrue(torch.all(errors > 0))

if __name__ == "__main__":
    unittest.main()