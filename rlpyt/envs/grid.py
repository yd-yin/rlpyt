import gym
import numpy as np


class Grid(gym.Env):
    def __init__(self, grid_size=5, max_steps=1, obs_inflation=1):
        super().__init__()
        self.grid_size = grid_size
        self.obs_inflation = obs_inflation
        self.obs_size = self.grid_size * self.obs_inflation
        self.max_steps = max_steps
        self.steps = 0
        self.goal_row = 0
        self.goal_col = 0
        self.action_space = gym.spaces.Discrete(self.grid_size * self.grid_size)
        self.observation_space = gym.spaces.Box(low=0.0,
                                                high=1.0,
                                                shape=(1, self.obs_size, self.obs_size),
                                                dtype=np.float32)

    def get_obs(self):
        obs = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
        obs[(self.goal_row * self.obs_inflation):((self.goal_row + 1) * self.obs_inflation),
            (self.goal_col * self.obs_inflation):((self.goal_col + 1) * self.obs_inflation)] = 1.0
        obs = np.expand_dims(obs, axis=0)
        return obs

    def step(self, action):
        assert 0 <= action < self.grid_size * self.grid_size
        self.steps += 1
        observation = self.get_obs()
        reward = 0.0
        done = False
        info = {}

        row = action // self.grid_size
        col = action % self.grid_size
        success = self.goal_row == row and self.goal_col == col

        if success:
            done = True
            reward = 1.0

        if self.steps >= self.max_steps:
            done = True
        
        return observation, reward, done, info

    def reset(self):
        self.goal_row = np.random.choice(self.grid_size)
        self.goal_col = np.random.choice(self.grid_size)
        self.steps = 0
        observation = self.get_obs()
        return observation

if __name__ == "__main__":
    env = Grid(grid_size=4, max_steps=1, obs_inflation=32)
    for i in range(10):
        print('episode [{i}]')
        env.reset()
        while True:
            obs, reward, done, info = env.step(0)
            print(reward)
            if done:
                break