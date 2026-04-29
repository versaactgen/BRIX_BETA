import argparse
import numpy as np
import gymnasium as gym
import torch
import torch.optim as optim
from torch.distributions import Categorical, Normal

from brix.envs.wrappers import make_env
from brix.models.mlp import PolicyValueMLP
from brix.buffers.replay_buffer import RolloutBuffer
from brix.algorithms.base import BaseAgent

class PPO(BaseAgent):
    """
    Proximal Policy Optimization (PPO) class inheriting from BaseAgent.
    """
    def __init__(
        self,
        env,
        model,
        buffer,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 3e-4,
        batch_size: int = 2048,
    ):
        super().__init__(env, model, buffer)
        self.gamma = gamma
        self.lam = lam
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.is_continuous = isinstance(self.env.action_space, gym.spaces.Box)

    def select_action(self, state):
        """
        Inference logic: returns an action, its log probability, and the state value.
        """
        state_tensor = torch.as_tensor(state, dtype=torch.float32).view(1, -1)

        with torch.no_grad():
            policy_output, value = self.model(state_tensor)

            if self.is_continuous:
                low = torch.as_tensor(self.env.action_space.low, dtype=torch.float32)
                high = torch.as_tensor(self.env.action_space.high, dtype=torch.float32)
                action_scale = torch.clamp((high - low) / 2.0, min=1e-6)
                action_bias = (high + low) / 2.0

                std = self.model.log_std.exp().expand_as(policy_output)
                dist = Normal(policy_output, std)
                raw_action = dist.sample()
                squashed_action = torch.tanh(raw_action)
                action_tensor = squashed_action * action_scale + action_bias

                log_prob = dist.log_prob(raw_action)
                log_prob -= torch.log(action_scale * (1 - squashed_action.pow(2)) + 1e-6)
                log_prob = log_prob.sum(dim=-1)

                action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            else:
                dist = Categorical(logits=policy_output)
                action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor)
                action = int(action_tensor.item())

        return action, float(log_prob.item()), float(value.squeeze(-1).item())

    def update(self):
        """
        The core PPO update step using the data in self.buffer.
        """
        data = self.buffer.get()

        return {
            "batch_size": int(data["rewards"].shape[0]),
            "reward_mean": float(data["rewards"].mean().item()),
            "reward_sum": float(data["rewards"].sum().item()),
            "done_count": int(data["dones"].sum().item()),
            "last_timestep": int(data["timesteps"][-1].item()),
        }

    def train_step(self):
        """
        Collect a rollout batch with env.step and store each timestep in the buffer.
        """
        state, _ = self.env.reset()

        for timestep in range(self.batch_size):
            action, log_prob, value = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.buffer.add(
                state=state,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=float(done),
                timestep=timestep,
            )

            state = next_state
            if done:
                state, _ = self.env.reset()

        return self.update()


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Argparse handles the CLI flags like --gamma and --lam
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="PPO Algorithm Base Code")
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="the id of the environment")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--lam", type=float, default=0.95, help="the lambda for GAE")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=2048, help="number of env steps to collect per rollout")
    parser.add_argument("--epochs", type=int, default=5, help="number of rollout epochs to run")
    args = parser.parse_args()

    print(f"Initializing PPO for {args.env_id}")
    print(
        f"Hyperparameters: gamma={args.gamma}, lam={args.lam}, "
        f"lr={args.lr}, batch_size={args.batch_size}"
    )

    # 1. Bring up the Environment
    env = make_env(args.env_id)
    is_continuous = isinstance(env.action_space, gym.spaces.Box)

    if not isinstance(env.observation_space, gym.spaces.Box):
        raise NotImplementedError("Only Box observation spaces are supported in this PPO scaffold.")

    input_dim = int(np.prod(env.observation_space.shape))

    if is_continuous:
        action_dim = int(np.prod(env.action_space.shape))
        action_shape = env.action_space.shape
    elif isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        action_shape = env.action_space.shape
    else:
        raise NotImplementedError("Only Box and Discrete action spaces are supported in this PPO scaffold.")

    # 2. Bring up the Model (MLP)
    model = PolicyValueMLP(
        input_dim=input_dim,
        action_dim=action_dim,
        is_continuous=is_continuous,
    )

    # 3. Bring up the Buffer
    buffer = RolloutBuffer(
        capacity=args.batch_size, 
        state_dim=env.observation_space.shape, 
        action_dim=action_shape,
    )

    # 4. Instantiate the Algorithm logic
    agent = PPO(
        env=env, 
        model=model, 
        buffer=buffer, 
        gamma=args.gamma, 
        lam=args.lam, 
        lr=args.lr,
        batch_size=args.batch_size,
    )

    # 5. Execute Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        metrics = agent.train_step()
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"batch={metrics['batch_size']}, reward_mean={metrics['reward_mean']:.3f}, "
            f"reward_sum={metrics['reward_sum']:.3f}, dones={metrics['done_count']}, "
            f"last_timestep={metrics['last_timestep']}"
        )
    
    print("Training process finished.")
