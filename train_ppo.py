from pathlib import Path
from time import perf_counter

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

TOTAL_STEPS = 1_000_000

# Create output folders beside this script.
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Create the environment without a game window.
env = gym.make("LunarLander-v3")

# Record the total reward and length of each episode.
env = Monitor(
    env,
    filename=str(LOG_DIR / "baseline.monitor.csv"),
)

# Create a new, untrained PPO agent.
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
    seed=42,
)

# Save a checkpoint every 100,000 environment steps.
checkpoint = CheckpointCallback(
    save_freq=100_000,
    save_path=str(MODEL_DIR / "checkpoints"),
    name_prefix="ppo_lander",
)

print(f"Starting training for approximately {TOTAL_STEPS:,} steps...")
start_time = perf_counter()

try:
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=checkpoint,
    )

    final_path = MODEL_DIR / "ppo_lunar_lander.zip"
    model.save(str(final_path))

    print("\nTraining finished.")
    print(f"Saved model: {final_path}")

except KeyboardInterrupt:
    partial_path = MODEL_DIR / "ppo_lunar_lander_interrupted.zip"
    model.save(str(partial_path))

    print("\nTraining stopped early.")
    print(f"Saved partial model: {partial_path}")

finally:
    env.close()
    elapsed_minutes = (perf_counter() - start_time) / 60
    print(f"Elapsed time: {elapsed_minutes:.2f} minutes")