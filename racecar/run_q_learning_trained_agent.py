import gymnasium as gym
import racecar_gym.envs.gym_api

import wrapper
from agents import DoubleQLearningRaceCarAgent


if __name__ == "__main__":
    track = 'circle'
    scenario = 'config/scenarios/' + track + '.yml'
    render_mode = "human"
    env = gym.make(
        id='SingleAgentRaceEnv-v0',
        scenario=scenario,
        vehicle_config_path='config/vehicles/racecar.yml',
        render_mode=render_mode
    )
    env.metadata['render_fps'] = 1
    env = wrapper.DiscreteActionWrapper(env, num_bins_motor=3, num_bins_steering=5)
    env = wrapper.DiscretizeObservationWrapper(env, lidar_bins=[0, 2.5, 5.0], num_velocity_bins=3, num_acceleration_bins=0)
    racecar_agent = DoubleQLearningRaceCarAgent(env)
    racecar_agent.load("trained_models/amit_double_q_table.pkl")
    observation, info = env.reset()
    done = False
    while not done:
        action = racecar_agent.get_action(observation, explore=False)

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
    env.close()
