import numpy as np
import torch

class RolloutBuffer:
    """
    A simple buffer to collect trajectories for on-policy algorithms like PPO.
    """
    def __init__(self, capacity: int, state_dim: tuple, action_dim: tuple):
        self.capacity = capacity
        
        # Initialize memory tensors
        self.states = np.zeros((capacity, *state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, value, log_prob, done):
        """Add a transition to the buffer."""
        if self.ptr >= self.capacity:
            raise RuntimeError("Buffer is full!")
            
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def get(self):
        """Retrieve collected trajectories and reset buffer."""
        assert self.ptr == self.capacity, "Buffer must be full before fetching"
        
        self.ptr = 0
        self.size = 0
        
        return dict(
            states=torch.tensor(self.states),
            actions=torch.tensor(self.actions),
            rewards=torch.tensor(self.rewards),
            values=torch.tensor(self.values),
            log_probs=torch.tensor(self.log_probs),
            dones=torch.tensor(self.dones)
        )


class ReplayBuffer:
    """
    A circular memory buffer suitable for off-policy algorithms like DQN or SAC.
    """
    def __init__(self, capacity: int, state_dim: tuple, action_dim: tuple):
        self.capacity = capacity
        
        self.states = np.zeros((capacity, *state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return dict(
            states=torch.tensor(self.states[indices]),
            actions=torch.tensor(self.actions[indices]),
            rewards=torch.tensor(self.rewards[indices]),
            next_states=torch.tensor(self.next_states[indices]),
            dones=torch.tensor(self.dones[indices])
        )
