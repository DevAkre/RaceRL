import torch
from stable_baselines3 import SAC

# ---- CONFIG ----
MODEL_PATH = "models/sac_best_model_circle.zip"
OUTPUT_PATH = "models/sac_policy_torch_weights.pth"
# ----------------

print(f"Loading SAC model from {MODEL_PATH}")
model = SAC.load(MODEL_PATH)

print("Extracting policy + Q networks...")
export = {
    "policy_state_dict": model.policy.state_dict(),
    "qf1_state_dict": model.policy.qf1.state_dict(),
    "qf2_state_dict": model.policy.qf2.state_dict(),
    "actor_state_dict": model.policy.actor.state_dict(),
}

print(f"Saving weights → {OUTPUT_PATH}")
torch.save(export, OUTPUT_PATH)

print("Export complete!")
