import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo


N_EPISODES = 20
FIRST_SEED = 1001  # Matches the previous evaluation: 1000 + episode number.
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ppo_lunar_lander.zip"

OUTCOMES = ("on_pad", "off_pad", "crash", "out_of_bounds", "timeout", "other_end")
OBS_NAMES = ("x", "y", "vx", "vy", "angle", "spin", "leg_1", "leg_2")


def describe_end(base, observation, terminated, truncated):
    """Read LunarLander-v3 physics state before resetting or closing it."""
    crashed = bool(base.game_over)
    out_of_bounds = abs(float(observation[0])) >= 1.0
    at_rest = not bool(base.lander.awake)
    # At a terrain join, a leg can have several contacts. Its observation flag
    # may become false when just one contact ends, while another still touches.
    # Count actual Box2D contacts instead of trusting that single boolean.
    leg_contacts = [
        any(
            edge.other == base.moon and edge.contact.touching and edge.contact.enabled
            for edge in leg.contacts
        )
        for leg in base.legs
    ]
    both_legs = len(leg_contacts) == 2 and all(leg_contacts)
    centre_x = float(base.lander.position.x)
    pad_left = float(base.helipad_x1)
    pad_right = float(base.helipad_x2)
    centre_in_pad = pad_left <= centre_x <= pad_right

    # A real terminal event takes precedence if both ending flags are true.
    if crashed:
        outcome = "crash"
    elif out_of_bounds:
        outcome = "out_of_bounds"
    elif terminated and at_rest and both_legs:
        outcome = "on_pad" if centre_in_pad else "off_pad"
    elif truncated and not terminated:
        outcome = "timeout"
    else:
        outcome = "other_end"

    return {
        "outcome": outcome,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "body_ground_contact": crashed,
        "at_rest": at_rest,
        "both_legs_touching": both_legs,
        "leg_1_physics_contact": leg_contacts[0],
        "leg_2_physics_contact": leg_contacts[1],
        "centre_in_pad": centre_in_pad,
        "body_centre_x_world": centre_x,
        "pad_left_x_world": pad_left,
        "pad_right_x_world": pad_right,
    }


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def save_summary(rows, run_dir, requested, status, metadata):
    """Use every completed, recorded episode, including poor outcomes."""
    counts = Counter(row["outcome"] for row in rows)
    scores = np.asarray([row["reward"] for row in rows], dtype=float)
    n = len(rows)
    summary = {
        "run_status": status,
        "requested_episodes": requested,
        "completed_episodes": n,
        "mean_reward": float(scores.mean()) if n else None,
        "reward_std_population": float(scores.std()) if n else None,
        "episodes_scoring_at_least_200": int((scores >= 200).sum()),
        "outcome_counts": {name: counts[name] for name in OUTCOMES},
        "on_pad_rate_percent": 100.0 * counts["on_pad"] / n if n else None,
        "metadata": metadata,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "BASELINE RECORDING RESULTS",
        f"Run status: {status}",
        f"Completed episodes: {n}/{requested}",
    ]
    if n:
        lines += [
            f"Mean reward: {scores.mean():.2f}",
            f"Reward standard deviation: {scores.std():.2f}",
            f"Episodes scoring at least 200: {int((scores >= 200).sum())}/{n}",
        ]
        lines += [f"{name}: {counts[name]}/{n}" for name in OUTCOMES]
        lines += [f"On-pad landing rate under our rule: {100 * counts['on_pad'] / n:.1f}%"]
    lines += [
        "",
        "Rule: at rest + both legs grounded + body centre between flags;",
        "body-ground crashes and out-of-bounds events take precedence.",
        "Review other_end and borderline landings in the matching videos.",
        "Only completed, fully recorded episodes enter the summary.",
    ]
    report = "\n".join(lines) + "\n"
    (run_dir / "summary.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Output folder: {run_dir}")
    return summary


def record_run(model, model_path, run_dir, n_episodes=N_EPISODES, first_seed=FIRST_SEED):
    """Record a fixed policy; this function never calls model.learn or model.save."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot = run_dir / "baseline_model.zip"
    shutil.copy2(model_path, snapshot)
    metadata = {
        "environment": "LunarLander-v3",
        "deterministic_actions": True,
        "first_seed": first_seed,
        "model_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "versions": {
            name: package_version(name)
            for name in ("gymnasium", "stable-baselines3", "torch", "Box2D", "moviepy")
        },
        "landing_rule": (
            "No body-ground crash or out-of-bounds event; naturally terminated; "
            "lander at rest; both legs grounded according to touching, enabled Box2D "
            "contacts with the terrain; body centre within inclusive flag bounds. "
            "This does not require the entire lander footprint to lie inside the flags."
        ),
    }
    video_dir = run_dir / "videos"
    base = gym.make("LunarLander-v3", render_mode="rgb_array")
    try:
        env = RecordVideo(
            base,
            video_folder=str(video_dir),
            episode_trigger=lambda episode_id: True,
            name_prefix="baseline",
            disable_logger=True,
        )
    except BaseException:
        base.close()
        raise

    episode_fields = [
        "episode", "seed", "reward", "steps", "video", "outcome", "terminated",
        "truncated", "body_ground_contact", "at_rest", "both_legs_touching",
        "leg_1_physics_contact", "leg_2_physics_contact", "centre_in_pad",
        "body_centre_x_world", "pad_left_x_world", "pad_right_x_world",
    ]
    trace_fields = ["episode", "seed", "step", "action", "reward", *OBS_NAMES]
    rows = []
    status = "complete"
    print(f"Recording {n_episodes} episodes. The game window will stay closed.")

    try:
        with (
            (run_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as episode_file,
            (run_dir / "steps.csv").open("w", newline="", encoding="utf-8") as trace_file,
        ):
            episode_writer = csv.DictWriter(episode_file, fieldnames=episode_fields)
            trace_writer = csv.DictWriter(trace_file, fieldnames=trace_fields)
            episode_writer.writeheader()
            trace_writer.writeheader()

            for episode in range(1, n_episodes + 1):
                seed = first_seed + episode - 1
                observation, info = env.reset(seed=seed)
                total_reward = 0.0
                steps = 0
                terminated = truncated = False
                trace_writer.writerow({
                    "episode": episode, "seed": seed, "step": 0, "action": "",
                    "reward": "", **dict(zip(OBS_NAMES, map(float, observation))),
                })

                while not (terminated or truncated):
                    action, _ = model.predict(observation, deterministic=True)
                    action = int(action)
                    observation, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    total_reward += float(reward)
                    trace_writer.writerow({
                        "episode": episode, "seed": seed, "step": steps,
                        "action": action, "reward": float(reward),
                        **dict(zip(OBS_NAMES, map(float, observation))),
                    })

                ending = describe_end(env.unwrapped, observation, terminated, truncated)
                # Finalize each full clip now, so it can be opened immediately.
                env.stop_recording()
                video_name = f"episode_{episode:02d}_seed_{seed}.mp4"
                (video_dir / f"baseline-episode-{episode - 1}.mp4").rename(
                    video_dir / video_name
                )
                row = {
                    "episode": episode, "seed": seed, "reward": total_reward,
                    "steps": steps, "video": f"videos/{video_name}", **ending,
                }
                episode_writer.writerow(row)
                episode_file.flush()
                trace_file.flush()
                rows.append(row)
                print(
                    f"Episode {episode:02d}/{n_episodes} | Seed {seed} | "
                    f"Reward {total_reward:.2f} | Steps {steps} | {ending['outcome']}",
                    flush=True,
                )

    except KeyboardInterrupt:
        status = "interrupted"
        print("\nStopped early. The summary includes completed recordings only.")
    except BaseException:
        status = "error"
        raise
    finally:
        try:
            env.close()  # Also writes any interrupted partial clip under its raw name.
        finally:
            summary = save_summary(rows, run_dir, n_episodes, status, metadata)
    return summary


def main():
    # Import here so the counting functions can be checked independently.
    from stable_baselines3 import PPO

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}\n"
            "Place record_baseline.py beside train_ppo.py, with models beside both."
        )
    model = PPO.load(str(MODEL_PATH), device="cpu")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "recordings" / f"baseline_{stamp}"
    record_run(model, MODEL_PATH, output)


if __name__ == "__main__":
    main()
