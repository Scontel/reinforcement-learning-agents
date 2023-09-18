import gym
from gym import spaces
import numpy as np

class SimpleGridEnv(gym.Env):
    def __init__(self, grid_size=5):
        super(SimpleGridEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4) # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)
        self.state = None
        self.goal = np.array([grid_size-1, grid_size-1])
        
    def reset(self):
        self.state = np.array([0, 0])
        return self.state
        
    def step(self, action):
        if action == 0: # Up
            self.state[1] = min(self.state[1] + 1, self.grid_size - 1)
        elif action == 1: # Down
            self.state[1] = max(self.state[1] - 1, 0)
        elif action == 2: # Left
            self.state[0] = max(self.state[0] - 1, 0)
        elif action == 3: # Right
            self.state[0] = min(self.state[0] + 1, self.grid_size - 1)
            
        done = np.array_equal(self.state, self.goal)
        reward = 1.0 if done else -0.01
        
        return self.state, reward, done, {}
