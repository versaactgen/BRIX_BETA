import torch
import torch.nn as nn

class BaseAgent:
    """
    Base class for RL Agents. Defines the primary interface that 
    training loops expect.
    """
    def __init__(self, env, model, buffer):
        self.env = env
        self.model = model
        self.buffer = buffer
        
    def select_action(self, state):
        """
        Given a state, select an action using the current model policy.
        """
        raise NotImplementedError
        
    def update(self):
        """
        Updates the model parameters using the collected transition buffer.
        """
        raise NotImplementedError
        
    def train_step(self):
        """
        Executes one step or epoch of the training loop.
        """
        raise NotImplementedError
