import gymnasium as gym
import stable_baselines3 as sb3
import torch

# Check that the three main libraries import.
print("Gymnasium:", gym.__version__)
print("Stable-Baselines3:", sb3.__version__)
print("PyTorch:", torch.__version__)

# Check that Lunar Lander and its physics engine work.
env = gym.make("LunarLander-v3")
observation, info = env.reset(seed=42)

# Perform one random action.
action = env.action_space.sample()
env.step(action)

print("Observation shape:", observation.shape)
print("Number of actions:", env.action_space.n)

env.close()
print("Setup successful!")