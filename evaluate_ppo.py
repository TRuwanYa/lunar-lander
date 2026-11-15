from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

N_EPISODES = 20

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ppo_lunar_lander.zip"

# Load the weights learned during training.
model = PPO.load(str(MODEL_PATH), device="cpu")

# Use the original environment and display the game.
env = gym.make("LunarLander-v3", render_mode="human")

episode_rewards = []

print(f"Testing the saved model over {N_EPISODES} episodes...")

try:
    for episode in range(1, N_EPISODES + 1):

        # Different fixed starting conditions for each episode.
        observation, info = env.reset(seed=1000 + episode)

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):

            # Let the trained model choose the action.
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = env.step(
                int(action)
            )

            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)

        print(
            f"Episode {episode}/{N_EPISODES} | "
            f"Reward: {total_reward:.2f} | Steps: {steps} | "
            f"Terminated: {terminated} | Truncated: {truncated}"
        )

except KeyboardInterrupt:
    print("\nEvaluation stopped early.")

finally:
    env.close()

# Summarize only the episodes that finished.
if episode_rewards:
    scores = np.array(episode_rewards)

    print("\nEvaluation results")
    print(f"Completed episodes: {len(scores)}")
    print(f"Mean reward: {scores.mean():.2f}")
    print(f"Reward standard deviation: {scores.std():.2f}")
    print(f"Lowest reward: {scores.min():.2f}")
    print(f"Highest reward: {scores.max():.2f}")
    print(
        f"Episodes scoring at least 200: "
        f"{int((scores >= 200).sum())}/{len(scores)}"
    )