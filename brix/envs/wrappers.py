import gymnasium as gym

class PreprocessWrapper(gym.Wrapper):
    """
    A simple customizable environment wrapper template.
    Usually we use wrappers to normalize observations, rewards, or handle action scaling.
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Custom reward shaping or obs transformation can happen here
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Custom obs transformation can happen here
        return obs, info

def make_env(env_name: str, seed: int = 0) -> gym.Env:
    """
    Factory function to create and wrap an environment.
    """
    env = gym.make(env_name)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    # Apply wrappers
    env = PreprocessWrapper(env)
    return env
