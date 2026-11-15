import gymnasium as gym

EPISODES = 5
PRINT_EVERY = 50

# Create Lunar Lander with a visible game window.
env = gym.make("LunarLander-v3", render_mode="human")

try:
    for episode in range(1, EPISODES + 1):

        # Start a new landing attempt.
        observation, info = env.reset()

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        print(f"\n--- Episode {episode} ---")
        print("Starting observation:", observation.round(3))

        while not (terminated or truncated):

            # Choose randomly:
            # 0 = do nothing
            # 1 = fire left engine
            # 2 = fire main engine
            # 3 = fire right engine
            action = env.action_space.sample()

            # Apply the action and advance the simulation one step.
            observation, reward, terminated, truncated, info = env.step(action)

            steps += 1
            total_reward += reward

            # Print occasionally, and always print the final step.
            if steps % PRINT_EVERY == 0 or terminated or truncated:
                print(
                    f"Step: {steps} | Action: {action} | "
                    f"Step reward: {reward:.2f}"
                )
                print("Observation:", observation.round(3))

        print(f"Episode finished after {steps} steps.")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Terminated: {terminated} | Truncated: {truncated}")

except KeyboardInterrupt:
    print("\nStopped by you.")

finally:
    env.close()