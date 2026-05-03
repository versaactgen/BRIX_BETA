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


```text
BRIX_BETA/
├── README.md                # Main project documentation
├── setup.py                 # Package installation script
├── requirements.txt         # Top-level dependency list
├── index.html               # Project homepage (GitHub Pages)
├── brix/                    # The primary Python package
│   ├── __init__.py
│   ├── algorithms/          # Implementations of RL algorithms
│   │   ├── __init__.py
│   │   ├── base.py          # Unified base class for all agents
│   │   └── ppo_base.py      # Proximal Policy Optimization
│   ├── buffers/             # Experience storage (Rollout / Replay)
│   │   ├── __init__.py
│   │   └── replay_buffer.py
│   ├── envs/                # Environment setup, wrappers & vectorization
│   │   ├── __init__.py
│   │   └── wrappers.py
│   └── models/              # Neural network architectures
│       ├── __init__.py
│       └── mlp.py           # Policy / Value MLP networks
└── tests/                   # Unit tests
    └── __init__.py
```


## Installation

### Prerequisites

Install [**uv**](https://docs.astral.sh/uv/getting-started/installation/) (a fast Python package & environment manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # Linux / macOS
```

### Setup

```bash
git clone https://github.com/versaactgen/BRIX_BETA.git
cd BRIX_BETA

# Create & activate a uv virtual environment
uv venv                     # creates .venv/ with the default Python (≥ 3.10)
source .venv/bin/activate   # activate the environment

# Install the package and all dependencies
uv pip install -e .
```

Top-level dependencies are tracked in `requirements.txt`, and `setup.py` reads from that file so the package metadata and the dependency list stay in sync.

### CUDA (optional)

If you want NVIDIA GPU acceleration for PyTorch, replace the default `torch` wheel with the official CUDA wheel that matches your driver. For Linux, PyTorch's stable selector currently offers CUDA 11.8, 12.6, and 12.8 wheels.

```bash
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Quick verification

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

---

## Quickstart

### Running PPO from the terminal

The built-in PPO script is runnable directly as a module:

```bash
# Make sure the environment is activated
source .venv/bin/activate

# Run PPO on CartPole (discrete) with default hyperparameters
python -m brix.algorithms.ppo_base

# Run PPO on HalfCheetah (continuous) with custom hyperparameters
python -m brix.algorithms.ppo_base \
    --env-id HalfCheetah-v5 \
    --gamma 0.99 \
    --lam 0.95 \
    --lr 3e-4 \
    --batch-size 4096 \
    --epochs 10
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--env-id` | `str` | `CartPole-v1` | Gymnasium environment ID (e.g. `HalfCheetah-v5`, `Ant-v5`) |
| `--gamma` | `float` | `0.99` | Discount factor for future rewards |
| `--lam` | `float` | `0.95` | Lambda for Generalized Advantage Estimation (GAE) |
| `--lr` | `float` | `3e-4` | Learning rate for the Adam optimizer |
| `--batch-size` | `int` | `2048` | Number of environment steps collected per rollout |
| `--epochs` | `int` | `5` | Number of rollout → update cycles to execute |

### Expected output

```text
Initializing PPO for CartPole-v1
Hyperparameters: gamma=0.99, lam=0.95, lr=0.0003, batch_size=2048
Starting training...
Epoch 1/5: batch=2048, reward_mean=1.000, reward_sum=2048.000, dones=93, last_timestep=2047
Epoch 2/5: batch=2048, reward_mean=1.000, reward_sum=2048.000, dones=87, last_timestep=2047
...
Training process finished.
```

### Using BRIX as a library

You can also import and compose the modules programmatically:

```python
from brix.envs.wrappers import make_env
from brix.models.mlp import PolicyValueMLP
from brix.buffers.replay_buffer import RolloutBuffer
from brix.algorithms.ppo_base import PPO

# 1. Setup Environment
env = make_env("CartPole-v1")

# 2. Setup Neural Network
model = PolicyValueMLP(
    input_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
)

# 3. Setup Buffer
buffer = RolloutBuffer(
    capacity=2048,
    state_dim=env.observation_space.shape,
    action_dim=env.action_space.shape,
)

# 4. Initialize & Train
agent = PPO(env=env, model=model, buffer=buffer, gamma=0.99, lam=0.95)

for epoch in range(5):
    metrics = agent.train_step()
    print(f"Epoch {epoch+1}: reward_mean={metrics['reward_mean']:.3f}")
```

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
