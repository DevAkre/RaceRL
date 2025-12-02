# Soft Actor Critic Model - How to Run

1. Ensure requirements including python packages, scenarios, and vechicles are present.

2. Train the model: python sac_racecar_agent.py train --env "Your_track_here" --timesteps <training steps> --model-path "models/racecar_sac_model"

3. Plot evaluation of training: python sac_eval.py

4. Evaluate and play with your model: python sac_racecar_agent.py play