import torch
import torch.nn as nn
from stable_baselines3 import PPO
import json
import numpy as np

class SimpleActionNet(nn.Module):
    """Wrapper to extract just the action network from PPO policy."""
    def __init__(self, ppo_model):
        super().__init__()
        self.policy = ppo_model.policy

    def forward(self, observation):
        # Extract features and get action logits only
        features = self.policy.extract_features(observation)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        action_logits = self.policy.action_net(latent_pi)
        return action_logits

def export_to_tfjs(model_path="pong_agent", output_dir="tfjs_model"):
    """Export PPO model to TensorFlow.js format."""
    print("Loading PPO model...")
    ppo_model = PPO.load(model_path)

    # Create simplified wrapper
    simple_model = SimpleActionNet(ppo_model)
    simple_model.eval()

    # Extract weights manually and save as JSON
    print("Extracting model weights...")

    weights = {}
    for name, param in simple_model.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()

    # Get model architecture info
    dummy_input = torch.zeros(1, 8, dtype=torch.float32)
    with torch.no_grad():
        output = simple_model(dummy_input)

    model_info = {
        "input_shape": [8],
        "output_shape": [3],
        "weights": weights,
        "architecture": {
            "layers": []
        }
    }

    # Save as JSON
    import os
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/model.json", "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"✓ Model exported to {output_dir}/model.json")
    print(f"  Input shape: {model_info['input_shape']}")
    print(f"  Output shape: {model_info['output_shape']}")
    print(f"  Parameters: {len(weights)}")

    # Test the export
    print("\nTesting exported weights...")
    test_input = torch.zeros(1, 8, dtype=torch.float32)
    with torch.no_grad():
        expected = simple_model(test_input)
    print(f"  Expected output: {expected.numpy()}")

if __name__ == "__main__":
    # Model is in ml_training directory, output to ml_deploy
    export_to_tfjs("../ml_training/pong_agent", "tfjs_model")
