import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from f1_environment_PPO import make_env
from sb3_contrib import RecurrentPPO
import os

class LidarCNNExtractor(BaseFeaturesExtractor):
    """
    CNN feature extractor for lidar + other sensors
    Processes lidar with 1D convolutions, concatenates with other features
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # observation_space.shape[0] = 1099
        # lidar: 1080, velocity: 6, acceleration: 6, pose: 6, time: 1
        
        self.lidar_size = 1080
        self.other_size = 6 + 6 + 6 + 1  # 19
        
        # CNN for lidar (1D convolutions)
        # Input: (batch, 1, 1080) -> process like 1D image
        self.lidar_cnn = nn.Sequential(

            nn.Conv1d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            

            nn.Conv1d(32, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            

            nn.Conv1d(64, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample_lidar = torch.zeros(1, 1, self.lidar_size)
            cnn_output_size = self.lidar_cnn(sample_lidar).shape[1]
        
        # MLP for other sensors
        self.other_mlp = nn.Sequential(
            nn.Linear(self.other_size, 64),
            nn.ReLU()
        )
        
        # Combined features
        combined_size = cnn_output_size + 64
        self.combined = nn.Sequential(
            nn.Linear(combined_size, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split observations
        lidar = observations[:, :self.lidar_size]  # (batch, 1080)
        other = observations[:, self.lidar_size:]   # (batch, 19)
        
        # Process lidar with CNN
        # Add channel dimension: (batch, 1080) -> (batch, 1, 1080)
        lidar = lidar.unsqueeze(1)
        lidar_features = self.lidar_cnn(lidar)  # (batch, cnn_output_size)
        
        # Process other sensors with MLP
        other_features = self.other_mlp(other)  # (batch, 64)
        
        # Concatenate and combine
        combined = torch.cat([lidar_features, other_features], dim=1)
        return self.combined(combined)


def train_ppo_with_cnn(total_timesteps=500000):
    """
    Train PPO with CNN feature extractor for lidar
    """
   

    print("PPO TRAINING WITH CNN LIDAR PROCESSING")

    
    # Create environment
    print("\nCreating environment...")
    env = DummyVecEnv([lambda: make_env(render_mode=None,track="circle")])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create directories for logging and checkpoints
    log_dir = "logs_cnn"
    checkpoint_dir = "checkpoints_cnn"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create PPO with CNN
    print("\nCreating PPO with CNN feature extractor...")
    
    policy_kwargs = dict(
        features_extractor_class=LidarCNNExtractor, # Using my custom CNN feature extraction instead of default MLP
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[256], vf=[256])]
    )
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=5e-5,
        n_steps=2048, 
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=log_dir, # Log the values to Tensorboard to get the training info later graphically
        policy_kwargs=policy_kwargs # Pass the custom CNN feature extractor to the model
    )

    # Train
    print(f"\nTraining for {total_timesteps:,} timesteps...")
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save
    model.save(os.path.join(checkpoint_dir, "ppo_cnn_model_columbia"))
    env.save(os.path.join(checkpoint_dir, "vec_normalize_columbia.pkl"))
    
    print("\nTraining complete!")
    print(f"Model saved to: {checkpoint_dir}/ppo_cnn_model_columbia.zip")


if __name__ == "__main__":
    train_ppo_with_cnn(total_timesteps=50_000)

