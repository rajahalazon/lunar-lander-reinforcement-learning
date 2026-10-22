import gymnasium as gym

class TimePenaltyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward -= 0.01
        return observation, reward, terminated, truncated, info
