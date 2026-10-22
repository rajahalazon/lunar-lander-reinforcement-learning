import gymnasium as gym
from stable_baselines3 import PPO
from reward_wrapper import TimePenaltyWrapper

env = TimePenaltyWrapper(gym.make("LunarLander-v3"))
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=500_000)
model.save("lunar_lander_shaped")
env.close()
print("Reward-shaped training complete!")
