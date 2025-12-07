from pathlib import Path
import gymnasium as gym
import racecar_gym.envs.gym_api
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from dqn_wrapper import DQNWrapper
from agents.dqn_agent import DQNNetwork, RacecarReplayBuffer
from train_dqn_circle import num_bins_motor, num_bins_steering


def detailed_episode_diagnosis(env, online_net, num_episodes=5):
    """
    Detailed diagnosis of what the agent is actually doing.
    """
    print("\n" + "="*70)
    print("DETAILED EPISODE DIAGNOSIS")
    print("="*70)

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        step = 0

        # Track what happens
        actions_taken = []
        velocities = []
        rewards_breakdown = {'frame': 0, 'collision': 0, 'progress': 0, 'other': 0}

        print(f"\n--- Episode {episode + 1} ---")

        while not done and step < 1000:
            # Get action from agent
            if online_net is None:  # Random policy
                random_action_idx = np.random.randint(0, len(env.allowed_actions))
                action = env.allowed_actions[random_action_idx]
            else:
                with torch.no_grad():
                    action, q_values = online_net.select_action(torch.FloatTensor(state).to(device), epsilon=0.0, deterministic=True)

            #continuous_action = action_discretizer.discrete_to_continuous(discrete_action)
            actions_taken.append(action)

            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track velocity
            velocities.append(next_state[6])  # Forward velocity

            # Breakdown rewards (approximate)
            if abs(reward - (-100.0)) < 0.1:
                rewards_breakdown['collision'] += reward
            elif abs(reward - (-1.0)) < 0.1:
                rewards_breakdown['frame'] += reward
            elif reward > 10:
                rewards_breakdown['progress'] += reward
            else:
                rewards_breakdown['other'] += reward

            episode_reward += reward
            state = next_state
            step += 1

            # Print every 100 steps
            if step % 100 == 0:
                recent_velocity = np.mean(velocities[-100:])
                print(f"  Step {step}: Avg velocity={recent_velocity:.2f} m/s, "
                      f"Reward so far={episode_reward:.2f}")

        # Episode summary
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total steps: {step}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"\n  Reward breakdown:")
        for key, val in rewards_breakdown.items():
            if val != 0:
                print(f"    {key}: {val:.2f}")

        # Velocity analysis
        velocities = np.array(velocities)
        print(f"\n  Velocity statistics:")
        print(f"    Mean: {velocities.mean():.2f} m/s")
        print(f"    Max:  {velocities.max():.2f} m/s")
        print(f"    Min:  {velocities.min():.2f} m/s")
        print(f"    Std:  {velocities.std():.2f} m/s")

        if velocities.mean() < 1.0:
            print(f"  ⚠️  WARNING: Car barely moving! (avg speed < 1 m/s)")

        if velocities.max() < 5.0:
            print(f"  ⚠️  WARNING: Car never goes fast! (max speed < 5 m/s)")

        # Action analysis
        action_counts = np.bincount(actions_taken, minlength=len(env.allowed_actions))
        most_common_action = action_counts.argmax()
        action_freq = action_counts[most_common_action] / len(actions_taken)

        print(f"\n  Action distribution:")
        print(f"    Most common action: {most_common_action} ({action_freq*100:.1f}% of time)")

        if action_freq > 0.8:
            print(f"  ⚠️  WARNING: Agent using same action {action_freq*100:.1f}% of time!")
            action_details = env.allowed_actions[action_idx]
            print(f"      Action {action_idx} (motor={action_details[0]:+.2f}, "
                  f"steering={action_details[1]:+.2f}")

        print(f"    Action usage:")
        for action_idx in range(len(env.allowed_actions)):
            if action_counts[action_idx] > 0:
                pct = action_counts[action_idx] / len(actions_taken) * 100
                action_details = env.allowed_actions[action_idx]
                print(f"Action {action_idx} (motor={action_details[0]:+.1f}, ")
                print(f"steering={action_details[1]:+.1f}): {pct:>5.1f}%")

def diagnose_q_values(model, env, device):
    """Check if Q-values have collapsed"""

    print("\n" + "="*70)
    print("Q-VALUE DIAGNOSIS")
    print("="*70)

    state, info = env.reset()

    # Collect Q-values for 100 steps
    all_q_values = []

    for step in range(100):


        action, q_value = online_net.select_action(torch.FloatTensor(state).to(device), epsilon=1.0,
                                          deterministic=True)
        all_q_values.append(q_value.cpu().numpy())

        next_state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            state, info = env.reset()
        else:
            state = next_state

    all_q_values = np.array(all_q_values)

    # Statistics
    print(f"\nQ-value statistics (across {len(all_q_values)} states):")
    print(f"  Mean:  {all_q_values.mean():.4f}")
    print(f"  Std:   {all_q_values.std():.4f}")
    print(f"  Min:   {all_q_values.min():.4f}")
    print(f"  Max:   {all_q_values.max():.4f}")
    print(f"  Range: {all_q_values.max() - all_q_values.min():.4f}")

    # Per-action statistics
    print(f"\nPer-action Q-value statistics:")
    for action_idx in range(len(env.allowed_actions)):
        action_q_values = all_q_values[:, action_idx]
        action_details = env.allowed_actions[action_idx]
        print(f"  Action {action_idx} (motor={action_details[0]:+.1f}, "
              f"steering={action_details[1]:+.1f}): "
              f"mean={action_q_values.mean():>7.3f}, std={action_q_values.std():>6.3f}")

    # Check for collapse
    if all_q_values.std() < 0.01:
        print(f"\n⚠️  CRITICAL: Q-values have COLLAPSED!")
        print(f"   All Q-values are essentially the same.")
        print(f"   The network cannot distinguish between actions.")
    elif all_q_values.std() < 0.1:
        print(f"\n⚠️  WARNING: Q-values have very low variance.")
        print(f"   The network is struggling to learn action preferences.")
    else:
        print(f"\n✓ Q-values show reasonable variance.")

    # Check if one action dominates
    mean_q_per_action = all_q_values.mean(axis=0)
    best_action = mean_q_per_action.argmax()
    q_differences = mean_q_per_action.max() - mean_q_per_action

    print(f"\nAction preference:")
    print(f"  Best action: {best_action}")
    print(f"  Q-value differences from best:")
    for action_idx in range(len(env.allowed_actions)):
        print(f"    Action {action_idx}: {q_differences[action_idx]:.4f}")

    if q_differences.max() < 0.01:
        print(f"\n⚠️  All actions have nearly identical Q-values!")



# Run this diagnosis
# If you have a trained agent:
device =  torch.device("mps")
#best_model_path = Path("best_dqn_model/best_dqn_model.pth")
best_model_path = Path("best_dqn_model/final_dqn_model.pth")

def plot_rewards(payload):
    rewards = payload['episode_rewards']
    episodes = range(len(rewards))
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, alpha=0.3, label='Episode Rewards')
    n_episodes = len(rewards)
    # Add moving average for clearer trend
    if n_episodes >= 10:
        window = n_episodes // 10
    else:
        window = 1
    if len(rewards) > window:
        moving_avg = [sum(rewards[i:i+window])/window
                    for i in range(len(rewards)-window+1)]
        plt.plot(range(window-1, len(rewards)), moving_avg,
                label=f'{window}-Episode Moving Average', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_num_step_per_episode(payload):
    num_steps = payload['episode_steps_all']
    episodes = range(len(num_steps))
    plt.figure(figsize=(10, 6))
    plt.bar(episodes, num_steps, alpha=0.6, label='Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps')
    plt.title('Steps per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    track = 'circle_cw'
    scenario = 'config/scenarios/' + track + '.yml'
    render_mode = "rgb_array"  # Set to None to disable rendering
    env = gym.make(
        id='SingleAgentRaceEnv-v0',
        scenario=scenario,
        vehicle_config_path='config/vehicles/og_racecar.yml',
        render_mode=render_mode
    )
    env = DQNWrapper(
        env,
        num_bins_motor=num_bins_motor,
        num_bins_steering=num_bins_steering
    )
    state_dim = env.observation_space_dim
    num_actions = len(env.allowed_actions)
    payload = torch.load(best_model_path, weights_only=False, map_location=device)
    online_net = DQNNetwork(state_dim, num_actions).to(device)
    online_net.load_state_dict(payload['online_net_state_dict'])
    #detailed_episode_diagnosis(env, online_net, num_episodes=3)
    #plot_rewards(payload)
    diagnose_q_values(online_net, env, device)
    plot_num_step_per_episode(payload)

