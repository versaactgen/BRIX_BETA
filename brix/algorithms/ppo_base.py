import argparse
import torch
import torch.optim as optim

from brix.envs.wrappers import make_env
from brix.models.mlp import PolicyValueMLP
from brix.buffers.replay_buffer import RolloutBuffer
from brix.algorithms.base import BaseAgent

class PPO(BaseAgent):
    """
    Proximal Policy Optimization (PPO) class inheriting from BaseAgent.
    """
    def __init__(self, env, model, buffer, gamma: float = 0.99, lam: float = 0.95, lr: float = 3e-4):
        super().__init__(env, model, buffer)
        self.gamma = gamma
        self.lam = lam
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        """
        Inference logic: returns an action, its log probability, and the state value.
        """
        # NOTE: To be implemented. Example:
        # with torch.no_grad():
        #     action_logits, value = self.model(state)
        #     # Create distribution, sample action...
        pass

    def update(self):
        """
        The core PPO update step using the data in self.buffer.
        """
        # NOTE: To be implemented.
        # Fetch trajectory data: data = self.buffer.get()
        # Calculate advantages, PPO clip loss, Value loss...
        pass

    def train_step(self):
        """
        A single epoch: collect experience, then update.
        """
        # 1. Collect rollout data by stepping in the environment
        # 2. Add transitions to self.buffer
        # 3. Once buffer is full, call self.update()
        pass


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Argparse handles the CLI flags like --gamma and --lam
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="PPO Algorithm Base Code")
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="the id of the environment")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--lam", type=float, default=0.95, help="the lambda for GAE")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    args = parser.parse_args()

    print(f"Initializing PPO for {args.env_id}")
    print(f"Hyperparameters: gamma={args.gamma}, lam={args.lam}, lr={args.lr}")

    # 1. Bring up the Environment
    env = make_env(args.env_id)

    # 2. Bring up the Model (MLP)
    # Note: assuming a discrete action space as a placeholder (like CartPole)
    model = PolicyValueMLP(
        input_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n 
    )

    # 3. Bring up the Buffer
    buffer = RolloutBuffer(
        capacity=2048, 
        state_dim=env.observation_space.shape, 
        action_dim=env.action_space.shape
    )

    # 4. Instantiate the Algorithm logic
    agent = PPO(
        env=env, 
        model=model, 
        buffer=buffer, 
        gamma=args.gamma, 
        lam=args.lam, 
        lr=args.lr
    )

    # 5. Execute Training Loop
    print("Starting training...")
    epochs = 5
    for epoch in range(epochs):
        agent.train_step()
    
    print("Training process finished.")
