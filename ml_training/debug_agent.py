from reinforcement_training import PongModel, PongEnv
import numpy as np

print("="*60)
print("DEBUGGING AGENT BEHAVIOR")
print("="*60)

pong_model = PongModel.load_model("pong_agent")
test_env = PongEnv(opponent_type="static")

obs, _ = test_env.reset()
print(f"\nInitial observation: {obs}")
print(f"Observation shape: {obs.shape}")
print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

action_counts = {0: 0, 1: 0, 2: 0}
total_reward = 0

print("\nWatching first 100 steps:")
print(f"{'Step':<6} {'Action':<10} {'Ball Y':<10} {'Paddle Y':<10} {'Reward':<10}")
print("-"*60)

for step in range(100):
    action_raw, _ = pong_model.predict(obs, deterministic=True)
    action = np.asarray(action_raw).item()
    action_counts[action] += 1

    obs, reward, done, truncated, info = test_env.step(action)
    total_reward += reward

    ball_y = obs[1] * 600
    paddle_y = obs[4] * 600
    action_name = ["UP", "STAY", "DOWN"][action]

    if step < 20 or done:
        print(f"{step:<6} {action_name:<10} {ball_y:<10.1f} {paddle_y:<10.1f} {reward:<10.2f}")

    if done:
        print(f"\nGame ended at step {step}")
        print(f"Final score: Agent {info['agent_score']} - {info['opp_score']} Opponent")
        print(f"Total reward: {total_reward:.2f}")
        break

print(f"\nAction distribution:")
print(f"  UP (0):   {action_counts[0]} times ({action_counts[0]/sum(action_counts.values())*100:.1f}%)")
print(f"  STAY (1): {action_counts[1]} times ({action_counts[1]/sum(action_counts.values())*100:.1f}%)")
print(f"  DOWN (2): {action_counts[2]} times ({action_counts[2]/sum(action_counts.values())*100:.1f}%)")

print("\n" + "="*60)
print("DIAGNOSIS:")
if action_counts[1] > 80:
    print("❌ Agent is stuck doing STAY - not moving at all!")
elif action_counts[0] > 80 or action_counts[2] > 80:
    print("❌ Agent is stuck moving in one direction!")
else:
    print("✓ Agent is moving somewhat normally")

if total_reward < -50:
    print("❌ Agent is losing badly")
elif total_reward < 0:
    print("⚠ Agent is losing")
else:
    print("✓ Agent is winning!")
print("="*60)
