import gymnasium as gym
from stable_baselines3 import PPO
from fixed_wrapper import FixedRewardWrapper

env = FixedRewardWrapper(gym.make("LunarLander-v3"))
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=500_000)
model.save("lunar_lander_fixed")
env.close()
print("Fixed-reward training complete!")
