import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import torch
    from torch import nn
    import gymnasium as gym
    import racecar_gym.envs.gym_api

    from dqn.racecar_dqn import RacecarDQN, RacecarReplayBuffer
    from wrapper import  DQNObservationWrapper, DQNActionWrapper
    return (
        DQNActionWrapper,
        DQNObservationWrapper,
        RacecarDQN,
        RacecarReplayBuffer,
        gym,
        nn,
        np,
        torch,
    )


@app.cell
def _(DQNActionWrapper, DQNObservationWrapper, gym):
    track = 'circle_cw'
    scenario = 'config/scenarios/' + track + '.yml'
    render_mode = "rgb_array"
    env = gym.make(
        id='SingleAgentRaceEnv-v0',
        scenario=scenario,
        vehicle_config_path='config/vehicles/og_racecar.yml',
        render_mode=render_mode
    )
    env = DQNObservationWrapper(env)
    env = DQNActionWrapper(env, num_bins_motor=5, num_bins_steering=5)
    return (env,)


@app.cell
def _(env):
    num_actions = env.num_bins_motor * env.num_bins_steering
    state_dim = env.observation_space_dim
    return num_actions, state_dim


@app.cell
def _(RacecarDQN, num_actions, state_dim, torch):
    device = torch.device("mps")
    online_net = RacecarDQN(state_dim, num_actions).to(device)
    target_net = RacecarDQN(state_dim, num_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    return device, online_net, target_net


@app.cell
def _(RacecarReplayBuffer, nn, online_net, torch):
    optimizer = torch.optim.Adam(online_net.parameters(), lr=0.001)
    replay_buffer = RacecarReplayBuffer(capacity=100_000)
    mse_loss = nn.MSELoss()
    return mse_loss, optimizer, replay_buffer


@app.cell
def _(
    device,
    env,
    mse_loss,
    np,
    online_net,
    optimizer,
    replay_buffer,
    target_net,
    torch,
):
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    target_update_freq = 1000

    # Training loop
    num_episodes = 10
    epsilon = epsilon_start
    total_steps = 0

    for episode in range(num_episodes):
        state, info = env.reset()
        #state = mock_state  # Using mock state for demonstration
        episode_reward = 0
        done = False
    
        while not done:
            # Preprocess state for action selection
            #state_array = preprocessor.preprocess(state)
        
            # Select action
            action = online_net.select_action(torch.FloatTensor(state).to(device), epsilon)
        
            # Execute action (mock for demonstration)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # next_state = {k: np.random.randn(*v.shape) if isinstance(v, np.ndarray) 
            #              else np.random.rand() for k, v in state.items()}
            reward = np.random.randn()
            done = np.random.rand() < 0.01  # Random termination
        
            # Store transition
            replay_buffer.add(state, action, reward, next_state, done)
        
            # Training step
            if len(replay_buffer) >= batch_size:
                # Sample batch
                states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = \
                    replay_buffer.sample(batch_size)
            
                # Move to device
                states_batch = states_batch.to(device)
                actions_batch = actions_batch.to(device)
                rewards_batch = rewards_batch.to(device)
                next_states_batch = next_states_batch.to(device)
                dones_batch = dones_batch.to(device)
            
                # Compute current Q-values
                current_q_values = online_net(states_batch)
                current_q = current_q_values.gather(1, actions_batch.unsqueeze(1)).squeeze()
            
                # Compute target Q-values
                with torch.no_grad():
                    next_q_values = target_net(next_states_batch)
                    max_next_q = next_q_values.max(dim=1)[0]
                    target_q = rewards_batch + gamma * max_next_q * (1 - dones_batch)
            
                # Compute loss and update
                loss = mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10)
                optimizer.step()
        
            # Update target network
            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(online_net.state_dict())
                print(f"Target network updated at step {total_steps}")
        
            state = next_state
            episode_reward += reward
            total_steps += 1
    
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
        #if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}, Buffer: {len(replay_buffer)}")

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
