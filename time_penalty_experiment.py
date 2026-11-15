import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import fmean, pstdev
from time import perf_counter

import gymnasium as gym

from record_baseline import OUTCOMES, describe_end, package_version


ROOT = Path(__file__).resolve().parent
ENVIRONMENT = "LunarLander-v3"
TOTAL_STEPS = 1_000_000
TRAINING_SEED = 42
STEP_PENALTY = 0.05
N_EPISODES = 20
FIRST_EVALUATION_SEED = 1001


class TimePenalty(gym.RewardWrapper):
    """Subtract a small cost for each step, including the final step."""

    def __init__(self, env, penalty=STEP_PENALTY):
        super().__init__(env)
        self.penalty = float(penalty)

    def reward(self, reward):
        return float(reward) - self.penalty


def save_evaluation(rows, run_dir, requested, first_seed, status):
    """Save all completed episodes, including crashes and timeouts."""
    scores = [row["reward"] for row in rows]
    counts = Counter(row["outcome"] for row in rows)
    on_pad_steps = [row["steps"] for row in rows if row["outcome"] == "on_pad"]
    off_pad_steps = [row["steps"] for row in rows if row["outcome"] == "off_pad"]
    n = len(rows)
    summary = {
        "run_status": status,
        "evaluation_reward": "original LunarLander-v3 reward; no time penalty",
        "deterministic_actions": True,
        "first_seed": first_seed,
        "requested_episodes": requested,
        "completed_episodes": n,
        "mean_reward": fmean(scores) if n else None,
        "reward_std_population": pstdev(scores) if n else None,
        "episodes_scoring_at_least_200": sum(score >= 200 for score in scores),
        "outcome_counts": {name: counts[name] for name in OUTCOMES},
        "on_pad_rate_percent": 100.0 * counts["on_pad"] / n if n else None,
        "mean_steps_all_episodes": fmean(row["steps"] for row in rows) if n else None,
        "mean_steps_on_pad": fmean(on_pad_steps) if on_pad_steps else None,
        "mean_steps_off_pad": fmean(off_pad_steps) if off_pad_steps else None,
    }
    (run_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "TIME-PENALTY MODEL: ORIGINAL-REWARD EVALUATION",
        f"Run status: {status}",
        f"Completed episodes: {n}/{requested}",
    ]
    if n:
        lines += [
            f"Mean original reward: {summary['mean_reward']:.2f}",
            f"Reward standard deviation: {summary['reward_std_population']:.2f}",
            f"Episodes scoring at least 200: {summary['episodes_scoring_at_least_200']}/{n}",
        ]
        lines += [f"{name}: {counts[name]}/{n}" for name in OUTCOMES]
        lines += [
            f"On-pad landing rate under our rule: {summary['on_pad_rate_percent']:.1f}%",
            f"Mean steps, all episodes: {summary['mean_steps_all_episodes']:.1f}",
            "Mean steps, on-pad episodes: "
            + (f"{fmean(on_pad_steps):.1f}" if on_pad_steps else "n/a"),
            "Mean steps, off-pad episodes: "
            + (f"{fmean(off_pad_steps):.1f}" if off_pad_steps else "n/a"),
        ]
    lines += [
        "",
        "Rule: at rest + both legs grounded + body centre between flags;",
        "body-ground crashes and out-of-bounds events take precedence.",
        "Only completed evaluation episodes enter the summary.",
        "Fewer steps alone is not success: check landing and crash counts too.",
    ]
    report = "\n".join(lines) + "\n"
    (run_dir / "evaluation_summary.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    return summary


def evaluate(model, run_dir, n_episodes=N_EPISODES, first_seed=FIRST_EVALUATION_SEED):
    """Test without the reward wrapper, using the baseline's counting rule."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(ENVIRONMENT)  # Deliberately uses the original reward.
    rows = []
    status = "complete"
    print(f"\nTesting {n_episodes} episodes with ORIGINAL rewards. No game window.")
    try:
        with (run_dir / "evaluation_episodes.csv").open(
            "w", newline="", encoding="utf-8"
        ) as output:
            writer = None
            for episode in range(1, n_episodes + 1):
                seed = first_seed + episode - 1
                observation, _ = env.reset(seed=seed)
                total_reward = 0.0
                steps = 0
                terminated = truncated = False

                while not (terminated or truncated):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, _ = env.step(int(action))
                    total_reward += float(reward)
                    steps += 1

                # Inspect physics before reset, using exactly the baseline rule.
                ending = describe_end(env.unwrapped, observation, terminated, truncated)
                row = {
                    "episode": episode, "seed": seed,
                    "reward": total_reward, "steps": steps, **ending,
                }
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                output.flush()
                rows.append(row)
                print(
                    f"Episode {episode:02d}/{n_episodes} | Seed {seed} | "
                    f"Reward {total_reward:.2f} | Steps {steps} | {ending['outcome']}",
                    flush=True,
                )
    except KeyboardInterrupt:
        status = "interrupted"
        print("\nEvaluation interrupted; keeping completed episodes only.")
    except BaseException:
        status = "error"
        raise
    finally:
        try:
            env.close()
        finally:
            summary = save_evaluation(rows, run_dir, n_episodes, first_seed, status)
    return summary


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = ROOT / "experiments" / f"time_penalty_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    model_path = run_dir / "ppo_time_penalty.zip"
    metadata = {
        "environment": ENVIRONMENT,
        "policy": "MlpPolicy",
        "fresh_model": True,
        "training_seed": TRAINING_SEED,
        "requested_training_steps": TOTAL_STEPS,
        "step_penalty": STEP_PENALTY,
        "training_reward": "original reward minus step_penalty on every step",
        "ppo_settings": "standard defaults, CPU, one environment; same setup as baseline",
        "evaluation_episodes": N_EPISODES,
        "first_evaluation_seed": FIRST_EVALUATION_SEED,
        "versions": {
            name: package_version(name)
            for name in ("gymnasium", "stable-baselines3", "torch", "Box2D")
        },
        "training_status": "running",
    }
    metadata_path = run_dir / "experiment.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Monitor is OUTSIDE the reward wrapper, so ep_rew_mean includes the fee.
    env = Monitor(
        TimePenalty(gym.make(ENVIRONMENT)),
        filename=str(run_dir / "training.monitor.csv"),
    )
    print(f"Output folder: {run_dir}")
    print(f"Training a fresh PPO model for {TOTAL_STEPS:,} requested steps.")
    print(f"Training reward: original reward minus {STEP_PENALTY} per step.")
    print("The training ep_rew_mean includes this penalty.")
    model = None
    start = perf_counter()
    try:
        model = PPO("MlpPolicy", env, verbose=1, device="cpu", seed=TRAINING_SEED)
        checkpoint = CheckpointCallback(
            save_freq=100_000,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo_time_penalty",
        )
        model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoint)
        model.save(str(model_path))
        metadata["training_status"] = "complete"
        metadata["saved_model"] = model_path.name
        metadata["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    except KeyboardInterrupt:
        metadata["training_status"] = "interrupted"
        if model is not None:
            interrupted_path = run_dir / "ppo_time_penalty_interrupted.zip"
            model.save(str(interrupted_path))
            metadata["saved_model"] = interrupted_path.name
            print(f"\nTraining interrupted. Partial model saved: {interrupted_path}")
    except BaseException:
        metadata["training_status"] = "error"
        raise
    finally:
        metadata["training_minutes"] = (perf_counter() - start) / 60
        metadata["actual_training_steps"] = model.num_timesteps if model is not None else 0
        try:
            env.close()
        finally:
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if metadata["training_status"] != "complete":
        print("The full experiment did not finish; automatic evaluation was skipped.")
        return

    print(f"\nTraining finished in {metadata['training_minutes']:.2f} minutes.")
    print(f"Saved model: {model_path}")
    # Reload the actual saved model for the comparison.
    model = PPO.load(str(model_path), device="cpu")
    evaluate(model, run_dir)
    print(f"Output folder: {run_dir}")


if __name__ == "__main__":
    main()
