# BRIX RL Framework

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)

**BRIX** is a modular, transparent, and scalable Reinforcement Learning framework. Designed with simplicity and modularity in mind, it provides robust out-of-the-box algorithmic implementations alongside simple base classes to enable building complex architectures for your RL experiments.

## Architecture

The framework is divided into several composable modules:

- **`brix.envs`**: Environment settings, vectorized configurations, and gym wrappers to handle the state and reward pipelines.
- **`brix.models`**: Core neural network architectures, such as shared representation MLPs and decoupled Policy/Value networks.
- **`brix.buffers`**: Experience replay, rollout buffers, and trajectory management for off-policy and on-policy data collection.
- **`brix.algorithms`**: Standardized algorithms (e.g., PPO) inheriting from simple, unified base classes to provide clean training and inference loops.


BRIX_BETA/
├── README.md               # Main project documentation
├── setup.py                # Package installation script
├── brix/                   # The primary Python package
│   ├── __init__.py
│   ├── algorithms/         # Implementations of RL algorithms (e.g., PPO, DQN)
│   │   ├── __init__.py
│   │   └── base.py
│   ├── buffers/            # Experience storage (Replay Buffers, Rollout Buffers)
│   │   ├── __init__.py
│   │   └── replay_buffer.py
│   ├── envs/               # Environment setup, wrappers and vectorization
│   │   ├── __init__.py
│   │   └── wrappers.py
│   └── models/             # Neural network architectures (Policy MLPs, Value MLPs)
│       ├── __init__.py
│       └── mlp.py
└── tests/                  # Directory for unit testing framework components
    └── __init__.py


## Installation

To use (or modify) the framework locally:

```bash
git clone https://github.com/yourusername/brix_rl.git
cd brix_rl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

Top-level dependencies are tracked in `requirements.txt`, and `setup.py` installs from that file so the package metadata and the dependency list stay in sync.

If you want NVIDIA GPU acceleration for PyTorch, replace the default `torch` wheel with the official CUDA wheel that matches your driver. For Linux, PyTorch's stable selector currently offers CUDA 11.8, 12.6, and 12.8 wheels.

Example:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Quick verification:

```bash
python -c "import gymnasium as gym, mujoco, torch; print(torch.__version__, torch.cuda.is_available())"
python - <<'PY'
import gymnasium as gym
env = gym.make('HalfCheetah-v5')
obs, info = env.reset()
print(type(obs), len(obs))
env.close()
PY
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
