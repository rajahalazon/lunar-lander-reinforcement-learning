import gymnasium as gym
from stable_baselines3 import PPO
from hover_wrapper import HoverRewardWrapper

env = HoverRewardWrapper(gym.make("LunarLander-v3"))
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=500_000)
model.save("lunar_lander_hover")
env.close()
print("Hover-reward training complete!")
