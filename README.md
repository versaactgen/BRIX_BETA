# BRIX RL Framework

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)

**BRIX** is a modular, transparent, and scalable Reinforcement Learning framework. Designed with simplicity and modularity in mind, it provides robust out-of-the-box algorithmic implementations alongside simple base classes to enable building complex architectures for your RL experiments.

## Architecture

The framework is divided into several composable modules:

- **`brix.envs`**: Environment settings, vectorized configurations, and gym wrappers to handle the state and reward pipelines.
- **`brix.models`**: Core neural network architectures, such as shared representation MLPs and decoupled Policy/Value networks.
- **`brix.buffers`**: Experience replay, rollout buffers, and trajectory management for off-policy and on-policy data collection.
- **`brix.algorithms`**: Standardized algorithms (e.g., PPO) inheriting from simple, unified base classes to provide clean training and inference loops.

## Installation

To use (or modify) the framework locally:

```bash
git clone https://github.com/yourusername/brix_rl.git
cd brix_rl
pip install -e .
```

## Quickstart

Here's an abstracted example of how the framework components hook together:

```python
import gymnasium as gym
from brix.envs.wrappers import make_env
from brix.models.mlp import PolicyValueMLP
from brix.buffers.replay_buffer import RolloutBuffer
from brix.algorithms.base import BaseAgent

# 1. Setup Environment
env = make_env("CartPole-v1")

# 2. Setup Neural Network Architectures
model = PolicyValueMLP(
    input_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n
)

# 3. Setup Buffer for Experience Collection
buffer = RolloutBuffer(capacity=2048, state_dim=env.observation_space.shape, action_dim=env.action_space.shape)

# 4. Initialize Algorithm
agent = BaseAgent(env=env, model=model, buffer=buffer)

# 5. Train
for epoch in range(100):
    agent.train_step()
```

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
