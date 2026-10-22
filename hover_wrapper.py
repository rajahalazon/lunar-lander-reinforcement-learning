import gymnasium as gym

class HoverRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if (abs(obs[4]) < 0.15 and abs(obs[2]) < 0.2
                and abs(obs[3]) < 0.2 and not (terminated or truncated)):
            reward += 1.0
        return obs, reward, terminated, truncated, info
