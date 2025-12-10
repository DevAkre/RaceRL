"""
Comamnd :  python quick_view.py --model_path checkpoints_cnn/ppo_cnn_model.zip
"""

import gymnasium
import numpy as np
import argparse
import os
from time import sleep

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import racecar_gym.envs.gym_api

from f1_environment_PPO import make_env

def visualize_model(model_path, num_episodes=5, track='columbia', deterministic=True):
    """
    Visualize a trained model
    """
   
    # To ensure you have a trained model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"\nLoading model")    
    model = PPO.load(model_path)

    print("Model policy:")
    print(model.policy)

    # Select track and create environment
    if make_env is not None:
        #env = make_env(track="columbia", render_mode="human")
        env = make_env(track=track, render_mode="human")
    else:
        # Fallback: create environment directly
        track_envs = {
            'austria': 'SingleAgentAustria-v0',
            'circle': 'SingleAgentCircle-v0',
            'columbia': 'SingleAgentColumbia-v0',
        }
        
        env_id = track_envs.get(track, 'circle')
        env = gymnasium.make(env_id, render_mode="human")
    
    vec_env = DummyVecEnv([lambda: env])

    model_dir = os.path.dirname(model_path)
    #norm_path = os.path.join(model_dir, "vec_normalize.pkl")
    norm_path = os.path.join(model_dir, "vec_normalize_columbia.pkl")
    
    if os.path.exists(norm_path):
        try:
            vec_env = VecNormalize.load(norm_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            print(f"Loaded: {norm_path}")
        except Exception as e:
            print(f"Could not load normalization: {e}")
    else:
        print(f"No normalization found at {norm_path}")
    
    # Run episodes
    print(f"\nRunning {num_episodes} episodes...")
    
    episode_stats = []
    
    try:
        for episode in range(num_episodes):
            obs = vec_env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            while not done:
                # Get action from model
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = vec_env.step(action)
                
                episode_reward += reward[0]
                steps += 1
                
                if steps % 500 == 0:
                    progress = info[0].get('progress', 0)
                    print(f"Step {steps}: Progress {progress*100:.1f}%", end='\r')

                
                # Small delay for visualization
                sleep(0.01)
                
                if done[0]:
                    break
            
            # Episode summary
            
            progress = info[0]['progress']
            wall_collision = info[0].get('wall_collision', False)
            lap = info[0]['lap']
            episode_time = info[0]['time']
            
            episode_stats.append({
                'steps': steps,
                'reward': episode_reward,
                'progress': progress,
                'collision': wall_collision,
                'time': episode_time,
                'lap': lap
            })
            
            print(f"Steps: {steps}")
            print(f"Reward: {episode_reward:.2f}")
            print(f"Progress: {progress*100:.1f}%")
            print(f"Lap: {lap}")
            
            if progress >= 0.99:
                print("LAP COMPLETED!") 
            if wall_collision:
                print("Crashed into wall")


    except KeyboardInterrupt:
        print("Visualization stopped by user")
    
    finally:
        vec_env.close()
    
    # Summary statistics
    if episode_stats:
        print("SUMMARY STATISTICS")
        
        avg_steps = np.mean([s['steps'] for s in episode_stats])
        avg_reward = np.mean([s['reward'] for s in episode_stats])
        avg_progress = np.mean([s['progress'] for s in episode_stats])
        max_progress = np.max([s['progress'] for s in episode_stats])
        laps_completed = sum(1 for s in episode_stats if s['lap'] >= 1)
        avg_time = np.mean([s['time'] for s in episode_stats])
        completed_times = [s['time'] for s in episode_stats if s['lap'] >= 1]
        avg_completed_time = np.mean(completed_times) if completed_times else 0
        std_dev_completed_time = np.std(completed_times) if completed_times else 0
        
        print(f"\nEpisodes: {len(episode_stats)}")
        print(f"Average steps: {avg_steps:.0f}")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Average progress: {avg_progress*100:.1f}%")
        print(f"Best progress: {max_progress*100:.1f}%")
        print(f"Laps completed: {laps_completed}/{len(episode_stats)}")
        print(f"Average episode time: {avg_time:.2f} sec")
        print(f"Average standard deviation for completed laps: {std_dev_completed_time:.6f} sec")

        if completed_times:
            print(f"Average time for completed laps: {avg_completed_time:.2f} sec ± {std_dev_completed_time:.6f} sec")
        
        if laps_completed > 0:
            print(f"Success rate: {laps_completed/len(episode_stats)*100:.1f}%")
    


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trained RL model racing"
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to model .zip file'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=50,
        help='Number of episodes to run (default: 50)'
    )   
    
    args = parser.parse_args()
    
    visualize_model(
        model_path=args.model_path,
        num_episodes=args.episodes,
    )


if __name__ == "__main__":
    main()


