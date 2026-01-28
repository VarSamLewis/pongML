import numpy as np
from stable_baselines3 import PPO
from reinforcement_training import PongEnv
import time


def evaluate_model(model_path, opponent_type, num_games=100):
    """
    Evaluate a trained model against a specific opponent type.

    Args:
        model_path: Path to the saved PPO model
        opponent_type: Type of opponent ('static', 'basic', 'medium', 'hard')
        num_games: Number of games to play

    Returns:
        Dictionary with evaluation results
    """
    model = PPO.load(model_path)
    env = PongEnv(opponent_type=opponent_type)

    wins = 0
    losses = 0
    total_agent_score = 0
    total_opp_score = 0
    episode_lengths = []

    for game in range(num_games):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1

        agent_score = info['agent_score']
        opp_score = info['opp_score']

        total_agent_score += agent_score
        total_opp_score += opp_score
        episode_lengths.append(steps)

        if agent_score > opp_score:
            wins += 1
        else:
            losses += 1

        print(f"\r{opponent_type.upper()}: Game {game+1}/{num_games} | Win: {agent_score} - {opp_score} Lose | Running W/L: {wins}/{losses}", end="", flush=True)

    print()

    win_rate = (wins / num_games) * 100
    avg_agent_score = total_agent_score / num_games
    avg_opp_score = total_opp_score / num_games
    avg_episode_length = np.mean(episode_lengths)

    return {
        'opponent_type': opponent_type,
        'num_games': num_games,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_agent_score': avg_agent_score,
        'avg_opp_score': avg_opp_score,
        'avg_episode_length': avg_episode_length
    }


def print_results(results):
    """Print formatted evaluation results."""
    print("\n" + "=" * 70)
    print(f"OPPONENT: {results['opponent_type'].upper()}")
    print("=" * 70)
    print(f"Games Played:       {results['num_games']}")
    print(f"Wins / Losses:      {results['wins']} / {results['losses']}")
    print(f"Win Rate:           {results['win_rate']:.1f}%")
    print(f"Avg Agent Score:    {results['avg_agent_score']:.2f}")
    print(f"Avg Opponent Score: {results['avg_opp_score']:.2f}")
    print(f"Avg Episode Length: {results['avg_episode_length']:.1f} steps")
    print("=" * 70)


def compare_models(baseline_path, supervised_path, opponent_types, num_games=100):
    """
    Compare baseline and supervised models across all opponent types.

    Args:
        baseline_path: Path to baseline model (e.g., 'pong_agent.zip')
        supervised_path: Path to supervised model (e.g., 'pong_agent_post.zip')
        opponent_types: List of opponent types to test against
        num_games: Number of games per opponent type
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: Baseline vs Supervised Fine-Tuning")
    print("=" * 70)
    print(f"Baseline Model:    {baseline_path}")
    print(f"Supervised Model:  {supervised_path}")
    print(f"Games per opponent: {num_games}")
    print("=" * 70)

    all_results = {
        'baseline': {},
        'supervised': {}
    }

    for opponent in opponent_types:
        print(f"\n\nTesting against {opponent.upper()} opponent...")
        print("-" * 70)

        print("\nBaseline Model:")
        baseline_results = evaluate_model(baseline_path, opponent, num_games)
        all_results['baseline'][opponent] = baseline_results
        print_results(baseline_results)

        print("\nSupervised Model:")
        supervised_results = evaluate_model(supervised_path, opponent, num_games)
        all_results['supervised'][opponent] = supervised_results
        print_results(supervised_results)

        # Print comparison
        wr_diff = supervised_results['win_rate'] - baseline_results['win_rate']
        score_diff = supervised_results['avg_agent_score'] - baseline_results['avg_agent_score']

        print("\nCOMPARISON:")
        print(f"Win Rate Change:    {wr_diff:+.1f}% ({baseline_results['win_rate']:.1f}% -> {supervised_results['win_rate']:.1f}%)")
        print(f"Avg Score Change:   {score_diff:+.2f} ({baseline_results['avg_agent_score']:.2f} -> {supervised_results['avg_agent_score']:.2f})")
        print("-" * 70)

    # Final summary
    print("\n\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"{'Opponent':<12} | {'Baseline WR':<12} | {'Supervised WR':<14} | {'Change':<10}")
    print("-" * 70)

    for opponent in opponent_types:
        baseline_wr = all_results['baseline'][opponent]['win_rate']
        supervised_wr = all_results['supervised'][opponent]['win_rate']
        change = supervised_wr - baseline_wr

        change_str = f"{change:+.1f}%"
        print(f"{opponent.capitalize():<12} | {baseline_wr:>10.1f}%  | {supervised_wr:>12.1f}%  | {change_str:>9}")

    print("=" * 70)


if __name__ == '__main__':
    start = time.time()

    # Test configuration
    baseline_model = "pong_agent.zip"
    supervised_model = "pong_agent_post.zip"
    opponent_types = ["static", "basic", "medium", "hard"]
    games_per_opponent = 100

    # Run comparison
    compare_models(
        baseline_path=baseline_model,
        supervised_path=supervised_model,
        opponent_types=opponent_types,
        num_games=games_per_opponent
    )

    elapsed = time.time() - start
    print(f"\nTotal evaluation time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
