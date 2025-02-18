# Credit Card Fraud Detection

This repository presents a comprehensive approach to credit card fraud detection using both unsupervised and supervised methods. The project focuses on training Variational Autoencoders (VAEs) to model legitimate transactions and identify anomalies that may indicate fraudulent behavior. Two VAE models are provided—one trained using the classical ELBO loss (with KL annealing) and another using a simple Mean Squared Error (MSE) loss. In addition, a set of base supervised models is included for comparative purposes.

## Project Overview

- **VAE with ELBO Loss (`VAE_ELBO_loss.ipynb`)**  
  Trains a Variational Autoencoder using the ELBO loss (which combines a reconstruction term with a KL divergence term) along with KL annealing. The model is trained solely on legitimate transactions to learn a compressed representation of normal behavior. Anomalies are detected via reconstruction error thresholds determined by an F‑Beta strategy with high Beta to overemphasize the importance of recall over accuracy. Optimal hyperparameters were determined offline using Optuna.

- **VAE with MSE Loss (`VAE_MSE_loss.ipynb`)**  
  Implements a stochastic autoencoder that uses only mean squared error (MSE) loss for reconstruction. While performance is lower compared to the ELBO-based approach, it serves as a valuable baseline for understanding the impact of incorporating the KL term and latent space regularization.

- **Base Models (`Base_models.ipynb`)**  
  Contains experiments with classical supervised models (e.g., Random Forest, LightGBM) to benchmark against the unsupervised VAE approaches.

**Important:**  
In addition to installing the Python dependencies listed in `requirements.txt`, users on macOS must install **libomp** via Homebrew:
```bash
brew install libomp
```

## Results & Performance

- The VAE with ELBO Loss model achieves a recall of approximately 97.3% on fraudulent transactions with a specificity around 86%. This sacrifice is acceptable given the context of the problem as correctly identifying fraudulent transactions is of the utmost importance.

- The VAE with MSE Loss model provides a baseline reconstruction performance.

- The Base Models notebook compares these unsupervised approaches with classical supervised models (and one alternate unsupervised model). These models on average have better accuracy than the VAE models but are far worse when comparing recall.

- Evaluation plots (included in the notebooks) display reconstruction error histograms, Precision-Recall curves, and ROC curves. The ROC curves mark both the optimal F‑Beta (β=30) threshold and additional percentile-based thresholds (90th, 95th, 97.5th, and 99th percentiles) for further performance insights.

## Contact

If you have any questions or feedback, please feel free to reach out via email at [abencarrington@egmail.com](mailto:abencarrington@gmail.com).
