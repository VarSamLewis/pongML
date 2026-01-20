from training_script import PongModel, PongEnv
import time


if __name__ == '__main__':
    pong_model = PongModel.load_model("pong_agent")

    print("=" * 60)
    print("TESTING PHASE")
    print("=" * 60)

    num_test_games = 1000
    start = time.time()
    test_results = pong_model.test(num_games=num_test_games, opponent_type="basic")
    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    agent_wins = sum(1 for a, o in test_results if a > o)
    opponent_wins = sum(1 for a, o in test_results if o > a)
    draws = sum(1 for a, o in test_results if a == o)

    print(f"Games Played: {num_test_games}")
    print(f"Agent Wins: {agent_wins} ({agent_wins/num_test_games*100:.1f}%)")
    print(f"Opponent Wins: {opponent_wins} ({opponent_wins/num_test_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_test_games*100:.1f}%)")
    print(f"Time taken: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    print("\nDetailed Results:")
    for i, (agent, opp) in enumerate(test_results, 1):
        result = "WIN" if agent > opp else "LOSS" if agent < opp else "DRAW"
        print(f"  Game {i:2d}: Agent {agent} - {opp} Opponent [{result}]")

    print("=" * 60)
