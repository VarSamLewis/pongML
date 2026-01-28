import torch
import torch.nn as nn
from stable_baselines3 import PPO

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

def export_ppo_to_onnx(model_path="pong_agent", output_path="pong_agent.onnx"):
    """Export simplified action-only model for browser inference."""
    print("Loading PPO model...")
    ppo_model = PPO.load(model_path)

    # Create simplified wrapper
    simple_model = SimpleActionNet(ppo_model)
    simple_model.eval()

    # Dummy input
    dummy_input = torch.zeros(1, 8, dtype=torch.float32)

    print("Exporting to ONNX with opset 14 (browser compatible)...")

    # Use legacy exporter for better compatibility
    with torch.no_grad():
        torch.onnx.export(
            simple_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['observation'],
            output_names=['action_logits'],
            dynamic_axes={
                'observation': {0: 'batch_size'},
                'action_logits': {0: 'batch_size'}
            },
            dynamo=False  # Use legacy exporter instead of new dynamo-based one
        )

    print(f"✓ Model exported to {output_path}")
    print("\nTesting exported model...")

    # Verify the export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")

    # Test with onnxruntime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        test_input = dummy_input.numpy()
        result = session.run(None, {'observation': test_input})
        print(f"✓ ONNX inference test passed")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {result[0].shape}")
        print(f"  Action logits: {result[0]}")
    except Exception as e:
        print(f"⚠ ONNX runtime test failed: {e}")

if __name__ == "__main__":
    export_ppo_to_onnx("pong_agent", "pong_agent.onnx")
