import gymnasium as gym
from gymnasium import spaces
import numpy as np

class WorkflowEnv(gym.Env):
    """
    Simulated workflow environment.
    State = workflow complexity level
    Action = optimization strategy
    """

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(1,), dtype=np.float32
        )
        self.state = None

    def reset(self, seed=None):
        self.state = np.array([np.random.randint(1, 10)], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        reward = 0

        if action == 0:
            reward = -1  # inefficient
        elif action == 1:
            reward = 1   # acceptable
        else:
            reward = 2   # optimal

        self.state = self.state - 0.5
        done = self.state[0] <= 0

        return self.state, reward, done, False, {}
