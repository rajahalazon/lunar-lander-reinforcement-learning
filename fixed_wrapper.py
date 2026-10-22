import gymnasium as gym

class FixedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_stability = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_stability = self.stability(obs)
        return obs, info

    def stability(self, obs):
        return -(abs(obs[2]) + abs(obs[3]) + abs(obs[4]))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_stability = self.stability(obs)
        improvement = current_stability - self.previous_stability
        reward += 2.0 * improvement
        reward -= 0.01
        self.previous_stability = current_stability
        return obs, reward, terminated, truncated, info
