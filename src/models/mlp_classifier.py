"""MLP classifier 784 -> 256 -> 64 -> 10 used in the Bagian 2 experiment.

z_MLP is the 64-d post-ReLU activation right before the classification layer.
Weights follow nn.Linear convention: W_l has shape (out, in), so the forward
map is h_l = ReLU(W_l h_{l-1} + b_l) and the "transpose projection" is W_l^T.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C


class MLPClassifier(nn.Module):
    def __init__(self, hidden=C.HIDDEN, latent=C.LATENT_AE, n_classes=10, p_drop=0.2):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, latent)
        self.fc3 = nn.Linear(latent, n_classes)
        self.drop = nn.Dropout(p_drop)

    def features(self, x, return_hidden=False):
        h1 = F.relu(self.fc1(x.flatten(1)))
        z = F.relu(self.fc2(self.drop(h1)))
        return (z, h1) if return_hidden else z

    def forward(self, x):
        return self.fc3(self.drop(self.features(x)))

    @property
    def weights(self):
        return self.fc1.weight, self.fc2.weight, self.fc3.weight

    # --- Bagian 2, Tahap 2 & 3: top-down transpose projections (no training) ---
    @torch.no_grad()
    def project_class(self, y_onehot):
        """x_hat = sigmoid(W1^T ReLU(W2^T ReLU(W3^T y_c)))"""
        W1, W2, W3 = self.weights
        z_hat = F.relu(y_onehot @ W3)          # (B,10)@(10,64)  == (W3^T y)^T
        h_hat = F.relu(z_hat @ W2)             # (B,64)@(64,256)
        return torch.sigmoid(h_hat @ W1).view(-1, *C.IMG_SHAPE)

    @torch.no_grad()
    def project_latent(self, z):
        """x_hat = sigmoid(W1^T ReLU(W2^T z_MLP))"""
        W1, W2, _ = self.weights
        return torch.sigmoid(F.relu(z @ W2) @ W1).view(-1, *C.IMG_SHAPE)
