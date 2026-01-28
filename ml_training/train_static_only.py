from reinforcement_training import PongModel, PongEnv
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import time

print("=" * 60)
print("TRAINING AGAINST STATIC OPPONENT ONLY")
print("=" * 60)

start = time.time()

env = make_vec_env(lambda: PongEnv(opponent_type="static"), n_envs=4)
pong_model = PongModel(env, total_timesteps=500_000)

print("\nStarting training...")
pong_model.train()
pong_model.save("pong_agent_static")

print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

if pong_model.metrics_callback:
    print(f"Total Steps: {pong_model.metrics_callback.timesteps[-1]:,}")
    print(f"Total Games: {pong_model.metrics_callback.total_games:,}")
    print(f"Final Avg Reward: {pong_model.metrics_callback.ep_rewards[-1]:.2f}")
    print(f"Final Avg Episode Length: {pong_model.metrics_callback.ep_lengths[-1]:.1f}")
    print(f"Model saved to: pong_agent_static.zip")

    plt.switch_backend('Agg')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(pong_model.metrics_callback.timesteps, pong_model.metrics_callback.ep_rewards, linewidth=2)
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Average Reward")
    ax1.set_title("Training Against Static Opponent: Reward")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax1.legend()

    ax2.plot(pong_model.metrics_callback.timesteps, pong_model.metrics_callback.ep_lengths, linewidth=2, color='orange')
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Average Episode Length")
    ax2.set_title("Episode Length")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_static.png", dpi=150)
    print(f"Training plot saved to: training_static.png")
    plt.close()

elapsed = time.time() - start
print(f"Time taken: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

print("\n" + "=" * 60)
print("TESTING")
print("=" * 60)

test_results = pong_model.test(num_games=20, opponent_type="static")

agent_wins = sum(1 for a, o in test_results if a > o)
opponent_wins = sum(1 for a, o in test_results if o > a)

print(f"\nAgent Wins: {agent_wins}/20 ({agent_wins/20*100:.1f}%)")
print(f"Opponent Wins: {opponent_wins}/20 ({opponent_wins/20*100:.1f}%)")

print("\nDetailed Results:")
for i, (agent, opp) in enumerate(test_results, 1):
    result = "WIN" if agent > opp else "LOSS"
    print(f"  Game {i:2d}: Agent {agent} - {opp} Opponent [{result}]")

print("=" * 60)
