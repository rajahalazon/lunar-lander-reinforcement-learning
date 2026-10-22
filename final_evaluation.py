import csv
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

models = {
    "Original PPO": "lunar_lander_ppo",
    "Time Penalty": "lunar_lander_shaped",
    "Hover Reward": "lunar_lander_hover",
    "Corrected Reward": "lunar_lander_fixed",
}
N_EPISODES = 100
results = []
for name, filename in models.items():
    print(f"\nEvaluating {name}...")
    model = PPO.load(filename, device="cpu")
    env = gym.make("LunarLander-v3")
    scores, durations, landings, hover_counts = [], [], [], []
    for episode in range(N_EPISODES):
        obs, info = env.reset(seed=1000 + episode)
        done, score, steps, hovering, landed = False, 0, 0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            steps += 1
            if (abs(obs[2]) < 0.2 and abs(obs[3]) < 0.2
                    and abs(obs[4]) < 0.15 and obs[6] == 0
                    and obs[7] == 0 and not (terminated or truncated)):
                hovering += 1
            if terminated and reward >= 100:
                landed = True
            done = terminated or truncated
        scores.append(score)
        durations.append(steps)
        landings.append(int(landed))
        hover_counts.append(hovering)
        results.append({
            "model": name, "episode": episode + 1, "seed": 1000 + episode,
            "score": round(score, 3), "duration_frames": steps,
            "successful_landing": int(landed),
            "slow_upright_airborne_frames": hovering,
        })
    env.close()
    print(f"Average score: {np.mean(scores):.2f}")
    print(f"Landing success: {sum(landings)}/{N_EPISODES}")
    print(f"Success rate: {np.mean(landings)*100:.1f}%")
    print(f"Average duration: {np.mean(durations):.1f} frames")
    print(f"Average slow/upright airborne frames: {np.mean(hover_counts):.1f}")
with open("final_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print("\nEvaluation complete! Results saved to final_results.csv")
