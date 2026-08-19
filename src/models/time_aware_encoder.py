"""
Time-Aware Point-Wise Observation Encoder Module.

Transforms irregularly sampled EHR observations (values X, mask M, relative time T, and time delta DeltaT)
into continuous point-wise hidden representations h_t in R^64.

Architecture Pipeline:
    [T_t, DeltaT_t] -> TimeFeatureEncoder (Linear(2->16) -> ReLU -> Linear(16->16)) -> time_embedding_t (R^16)
    [X_t, M_t, time_embedding_t] (R^26) -> Encoder MLP (Linear(26->64) -> ReLU -> Linear(64->64)) -> h_t (R^64)
"""

import torch
import torch.nn as nn


class TimeFeatureEncoder(nn.Module):
    """
    Transforms numerical time features [T_t, DeltaT_t] into a learnable time representation.
    """

    def __init__(self, time_embed_dim: int = 16):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(2, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, T: torch.Tensor, DeltaT: torch.Tensor) -> torch.Tensor:
        """
        Args:
            T: Relative time tensor of shape (N,) or (B, N)
            DeltaT: Elapsed time gap tensor of shape (N,) or (B, N)

        Returns:
            Time embedding tensor of shape (N, time_embed_dim) or (B, N, time_embed_dim)
        """
        # Stack T and DeltaT along final dimension
        time_input = torch.stack([T, DeltaT], dim=-1).to(torch.float32)
        return self.net(time_input)


class TimeAwareEncoder(nn.Module):
    """
    Point-wise Time-Aware Encoder module.

    Encodes irregular EHR observation tuples (X, M, T, DeltaT) into hidden state representations h_t.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, time_embed_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim

        self.time_encoder = TimeFeatureEncoder(time_embed_dim=time_embed_dim)

        in_features = input_dim + input_dim + time_embed_dim  # 5 + 5 + 16 = 26
        self.encoder_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        X: torch.Tensor,
        M: torch.Tensor,
        T: torch.Tensor,
        DeltaT: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encodes observation tensors into hidden representation h_t.

        Args:
            X: Observation tensor of shape (N, 5) or (B, N, 5)
            M: Observation mask tensor of shape (N, 5) or (B, N, 5)
            T: Relative time tensor of shape (N,) or (B, N)
            DeltaT: Elapsed time gap tensor of shape (N,) or (B, N)

        Returns:
            Hidden representation tensor h_t of shape (N, 64) or (B, N, 64)
        """
        X = X.to(torch.float32)
        M = M.to(torch.float32)
        T = T.to(torch.float32)
        DeltaT = DeltaT.to(torch.float32)

        # 1. Encode time features (T_t, DeltaT_t) -> R^16
        time_embed = self.time_encoder(T, DeltaT)

        # 2. Concatenate [X_t, M_t, time_embedding_t] -> R^26
        concat_input = torch.cat([X, M, time_embed], dim=-1)

        # 3. Pass through Encoder MLP -> R^64
        h = self.encoder_mlp(concat_input)
        return h
