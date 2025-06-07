import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """
    Policy network for entity-level embedding (state + action concatenation).
    Used when each state is a (source, target) embedding and each action is a neighbor node.
    """
    def __init__(self, input_dim: int = 1152, hidden_dim: int = 256):
        """
        Args:
            input_dim: Concatenated size of state (768) and action (384)
            hidden_dim: Size of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a scalar score per action
        )

    def forward(self, state: torch.Tensor, action_vecs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the policy network.

        Args:
            state: [B, 768] tensor of state embeddings
            action_vecs: [B, N, 384] tensor of N candidate actions (neighbor embeddings)

        Returns:
            logits: [B, N] tensor of scores for each action
        """
        B, N, _ = action_vecs.shape
        state_expanded = state.unsqueeze(1).expand(-1, N, -1)     # [B, N, 768]
        x = torch.cat([state_expanded, action_vecs], dim=-1)      # [B, N, 1152]
        logits = self.net(x).squeeze(-1)                          # [B, N]
        return logits


class TriplePolicyNet(nn.Module):
    """
    Policy network that scores candidate triples directly using triple embeddings.
    Ignores the separate state vector and operates on semantic vector of (H, R, T) triples.
    """
    def __init__(self, triple_dim: int = 384, hidden_dim: int = 256):
        """
        Args:
            triple_dim: Dimensionality of each triple embedding
            hidden_dim: Hidden layer size
        """
        super().__init__()
        self.fc1 = nn.Linear(triple_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output a scalar score per triple

    def forward(self, state: torch.Tensor, triple_vecs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (note: `state` is ignored).

        Args:
            state: [1, D] dummy tensor (ignored)
            triple_vecs: [1, N, triple_dim] tensor of triple embeddings

        Returns:
            scores: [1, N] tensor of scores for each triple
        """
        x = triple_vecs.squeeze(0)           # [N, triple_dim]
        x = torch.relu(self.fc1(x))          # [N, hidden_dim]
        scores = self.fc2(x).squeeze(-1)     # [N]
        return scores.unsqueeze(0)           # [1, N] for batch-consistency
