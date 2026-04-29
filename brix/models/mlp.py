import torch
import torch.nn as nn

class MLPNetwork(nn.Module):
    """
    Standard Multi-Layer Perceptron used for generic base networks.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=(64, 64), activation=nn.ReLU):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
            
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class PolicyValueMLP(nn.Module):
    """
    A combined model mapping state to both an action distribution (policy) 
    and a state value estimate (value).
    """
    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_dims=(64, 64),
        is_continuous: bool = False,
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.is_continuous = is_continuous
        
        # Policy Network
        self.actor = MLPNetwork(input_dim, action_dim, hidden_dims)
        if self.is_continuous:
            self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
        
        # Value Network
        self.critic = MLPNetwork(input_dim, 1, hidden_dims)
        
    def forward(self, state):
        # Depending on action space, this might output logits for Categorical
        # or mean/std for Normal distributions.
        action_logits = self.actor(state)
        value = self.critic(state)
        return action_logits, value
