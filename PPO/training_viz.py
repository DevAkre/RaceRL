"""
PPO training visualization from TensorBoard logs
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os

def extract_tensorboard_data(log_path):
    """Extract metrics from TensorBoard log"""

    ea = event_accumulator.EventAccumulator(log_path)
    ea.Reload()
    
    print("\nAvailable metrics:")
    for tag in ea.Tags()['scalars']:
        print(f"{tag}")
    
    # Extract data
    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': np.array(steps), 'values': np.array(values)}
    
    return data


def plot_training_overview(data, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')
    
    colors = {
        'reward': '#2E86AB',
        'length': '#A23B72',
        'policy': '#F18F01'
    }

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, hspace=0.4)
    
    # Reward
    ax1 = fig.add_subplot(gs[0])
    if 'rollout/ep_rew_mean' in data:
        steps = data['rollout/ep_rew_mean']['steps']
        rewards = data['rollout/ep_rew_mean']['values']
        ax1.plot(steps, rewards, color=colors['reward'], linewidth=1.5)
        ax1.set_ylabel('Episode Reward', fontweight='bold', fontsize=11)
        ax1.set_title('Training Overview - PPO', 
                      fontsize=15, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
    
    # Episode length
    ax2 = fig.add_subplot(gs[1])
    if 'rollout/ep_len_mean' in data:
        steps = data['rollout/ep_len_mean']['steps']
        lengths = data['rollout/ep_len_mean']['values']
        ax2.plot(steps, lengths, color=colors['length'], linewidth=1.5)
        ax2.set_ylabel('Episode Length', fontweight='bold', fontsize=11)
        ax2.grid(True, alpha=0.3)
    
    # Policy loss
    ax3 = fig.add_subplot(gs[2])
    if 'train/policy_gradient_loss' in data:
        steps = data['train/policy_gradient_loss']['steps']
        policy_loss = data['train/policy_gradient_loss']['values']
        ax3.plot(steps, policy_loss, color=colors['policy'], linewidth=1.5)
        ax3.set_xlabel('Training Steps', fontweight='bold', fontsize=11)
        ax3.set_ylabel('Policy Loss', fontweight='bold', fontsize=11)
        ax3.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, 'training_overview.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/training_overview.png")
    plt.close()




if __name__ == "__main__":
    log_path = "logs_cnn/PPO_31"
    #track = "circle"
    track = "columbia"
    data = extract_tensorboard_data(log_path)
    plot_training_overview(data, save_dir=f'plots/{track}')    
    print(f"Generated: plots/{track}/training_overview.png")