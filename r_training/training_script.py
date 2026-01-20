import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import time as time


class PongEnv(gym.Env):
    """Custom Pong environment for RL training with 8-dimensional state space."""

    metadata = {"render_mode": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, opponent_type="basic"):
        """Initialize Pong environment with game parameters."""
        super().__init__()
        self.width = 800
        self.height = 600
        self.paddle_width = 10
        self.paddle_height = 25
        self.ball_radius = 8
        self.paddle_speed = 5
        self.ball_speed_initial = 5

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.opponent_type = opponent_type

    def reset(self, seed=None, options=None):
        """Reset environment to initial state and return observation."""
        super().reset(seed=seed)

        self.ball_x = self.width / 2
        self.ball_y = self.height / 2

        angle = self.np_random.uniform(-np.pi/4, np.pi/4)
        direction = self.np_random.choice([-1, 1])
        self.ball_vx = direction * self.ball_speed_initial * np.cos(angle)
        self.ball_vy = self.ball_speed_initial * np.sin(angle)

        self.agent_y = self.height / 2
        self.opp_y = self.height / 2

        self.agent_score = 0
        self.opp_score = 0

        self.steps = 0
        self.max_steps = 10000

        return self._get_obs(), {}

    def step(self, action):
        """Execute action, update state, and return observation, reward, done, truncated, info."""
        self.steps += 1

        if action == 0:
            self.agent_y = max(0, self.agent_y - self.paddle_speed)
        elif action == 2:
            self.agent_y = min(self.height - self.paddle_height, self.agent_y + self.paddle_speed)

        self._move_opponent()

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_y <= self.ball_radius or self.ball_y >= self.height - self.ball_radius:
            self.ball_vy *= -1
            self.ball_y = np.clip(self.ball_y, self.ball_radius, self.height - self.ball_radius)

        reward = 0.0
        terminated = False
        agent_scored = False
        opp_scored = False
        agent_hit_ball = False
        opp_hit_ball = False

        if self.ball_x <= self.paddle_width + self.ball_radius:
            if self.agent_y <= self.ball_y <= self.agent_y + self.paddle_height:
                agent_hit_ball = True
                self.ball_vx = abs(self.ball_vx)
                self.ball_x = self.paddle_width + self.ball_radius

                hit_pos = (self.ball_y - self.agent_y) / self.paddle_height - 0.5
                self.ball_vy += hit_pos * 3

                self.ball_vx *= 1.05
                self.ball_vy *= 1.05

                reward += 10.0
            else:
                opp_scored = True
                self.opp_score += 1
                terminated = True

        if self.ball_x >= self.width - self.paddle_width - self.ball_radius:
            if self.opp_y <= self.ball_y <= self.opp_y + self.paddle_height:
                opp_hit_ball = True
                self.ball_vx = -abs(self.ball_vx)
                self.ball_x = self.width - self.paddle_width - self.ball_radius

                hit_pos = (self.ball_y - self.opp_y) / self.paddle_height - 0.5
                self.ball_vy += hit_pos * 3
            else:
                agent_scored = True
                self.agent_score += 1
                terminated = True

        if agent_scored:
            reward += 20.0
        if opp_scored:
            reward -= 20.0

        paddle_center = self.agent_y + self.paddle_height / 2
        y_dist_to_ball = abs(paddle_center - self.ball_y) / self.height
        x_dist_factor = max(0.1, self.ball_x / self.width)
        reward += 0.01 * (1.0 - y_dist_to_ball) / x_dist_factor

        if opp_hit_ball:
            reward -= 1.0

        if self.steps >= self.max_steps:
            terminated = True

        return self._get_obs(), reward, terminated, False, {
            "agent_score": self.agent_score,
            "opp_score": self.opp_score
        }

    def _get_obs(self):
        """Return normalized observation vector."""
        obs = np.array([
            self.ball_x / self.width,
            self.ball_y / self.height,
            (self.ball_vx + 10) / 20,
            (self.ball_vy + 10) / 20,
            self.agent_y / self.height,
            (self.agent_y + self.paddle_height) / self.height,
            self.opp_y / self.height,
            (self.opp_y + self.paddle_height) / self.height,
        ], dtype=np.float32)
        return obs

    def _move_opponent(self):
        """Move opponent paddle based on opponent type."""
        if self.opponent_type == "static":
            pass
        elif self.opponent_type == "basic":
            paddle_center = self.opp_y + self.paddle_height / 2
            if paddle_center < self.ball_y - 5:
                self.opp_y = min(self.height - self.paddle_height, self.opp_y + self.paddle_speed)
            elif paddle_center > self.ball_y + 5:
                self.opp_y = max(0, self.opp_y - self.paddle_speed)
        elif self.opponent_type == "medium":
            if self.ball_vx > 0:
                paddle_center = self.opp_y + self.paddle_height / 2
                if paddle_center < self.ball_y - 5:
                    self.opp_y = min(self.height - self.paddle_height, self.opp_y + self.paddle_speed * 0.9)
                elif paddle_center > self.ball_y + 5:
                    self.opp_y = max(0, self.opp_y - self.paddle_speed * 0.9)
        elif self.opponent_type == "hard":
            paddle_center = self.opp_y + self.paddle_height / 2
            if paddle_center < self.ball_y - 2:
                self.opp_y = min(self.height - self.paddle_height, self.opp_y + self.paddle_speed * 1.2)
            elif paddle_center > self.ball_y + 2:
                self.opp_y = max(0, self.opp_y - self.paddle_speed * 1.2)

    def render(self):
        """Render environment (optional)."""
        return None

    def close(self):
        """Clean up resources."""
        pass

class TrainingMetricsCallback(BaseCallback):
    """Callback to track training metrics and display progress."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_rewards = []
        self.ep_lengths = []
        self.timesteps = []
        self.value_losses = []
        self.policy_losses = []
        self.total_games = 0

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])

            self.ep_rewards.append(mean_reward)
            self.ep_lengths.append(mean_length)
            self.timesteps.append(self.num_timesteps)
            self.total_games += len(self.model.ep_info_buffer)

            print(f"\rTraining: {self.num_timesteps:,} steps | Games: {self.total_games:,} | Avg Reward: {mean_reward:.2f}", end="", flush=True)
        return True

    def _on_training_end(self):
        print()


class PongModel:
    """Wrapper class for PPO model training and export."""

    def __init__(self, env, net_arch=[64, 64], learning_rate=3e-4, total_timesteps = 100_000):
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            verbose=0,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=net_arch)
        )
        self.metrics_callback = None
        self.total_timesteps = total_timesteps

    def train(self):
        self.metrics_callback = TrainingMetricsCallback()
        self.model.learn(total_timesteps=self.total_timesteps, callback=self.metrics_callback)

    def save(self, path="pong_agent"):
        self.model.save(path)

    def load(self, path="pong_agent"):
        self.model = PPO.load(path)

    @classmethod
    def load_model(cls, path="pong_agent"):
        """Load a trained model from file."""
        dummy_env = make_vec_env(lambda: PongEnv(), n_envs=1)
        pong_model = cls(dummy_env)
        pong_model.load(path)
        return pong_model

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)

    def export_to_onnx(self, output_path="pong_agent.onnx"):
        dummy_input = torch.zeros(1, 8)
        torch.onnx.export(
            self.model.policy.to("cpu"),
            dummy_input,
            output_path,
            input_names=["observation"],
            output_names=["action"]
        )

    def test(self, num_games=10, opponent_type="basic"):
        """Test the trained model and return results."""
        test_env = PongEnv(opponent_type=opponent_type)
        test_results = []

        for game in range(num_games):
            obs, _ = test_env.reset()
            done = False
            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = test_env.step(action)
            test_results.append((info['agent_score'], info['opp_score']))
            print(f"\rTesting: Game {game+1}/{num_games}", end="", flush=True)

        print()
        return test_results

    def plot_training_metrics(self, save_path="training_metrics.png"):
        """Generate training metrics plots."""
        if self.metrics_callback is None:
            print("No training metrics available")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(self.metrics_callback.timesteps, self.metrics_callback.ep_rewards, linewidth=2)
        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel("Average Reward")
        ax1.set_title("Training Progress: Episode Reward")
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.metrics_callback.timesteps, self.metrics_callback.ep_lengths, linewidth=2, color='orange')
        ax2.set_xlabel("Timesteps")
        ax2.set_ylabel("Average Episode Length")
        ax2.set_title("Training Progress: Episode Length")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Training metrics plot saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("PONG RL TRAINING - CURRICULUM LEARNING")
    print("=" * 60)
    start = time.time()

    total_timesteps = 1_000_000
    steps_per_difficulty = total_timesteps // 4
    difficulties = ["static", "basic", "medium", "hard"]

    all_rewards = []
    all_lengths = []
    all_timesteps = []

    env = make_vec_env(lambda: PongEnv(opponent_type="static"), n_envs=4)
    pong_model = PongModel(env, total_timesteps=steps_per_difficulty)

    for i, difficulty in enumerate(difficulties):
        print(f"\n{'='*60}")
        print(f"PHASE {i+1}/4: Training against '{difficulty}' opponent")
        print(f"Steps: {steps_per_difficulty:,}")
        print(f"{'='*60}")

        if i > 0:
            env = make_vec_env(lambda d=difficulty: PongEnv(opponent_type=d), n_envs=4)
            pong_model.model.set_env(env)
            pong_model.total_timesteps = steps_per_difficulty

        pong_model.train()

        if pong_model.metrics_callback:
            all_rewards.extend(pong_model.metrics_callback.ep_rewards)
            all_lengths.extend(pong_model.metrics_callback.ep_lengths)
            offset = i * steps_per_difficulty
            all_timesteps.extend([t + offset for t in pong_model.metrics_callback.timesteps])

            print(f"\nPhase {i+1} Summary:")
            print(f"  Final Avg Reward: {pong_model.metrics_callback.ep_rewards[-1]:.2f}")
            print(f"  Final Avg Length: {pong_model.metrics_callback.ep_lengths[-1]:.1f}")

    pong_model.save("pong_agent")

    print("\n" + "=" * 60)
    print("FINAL TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total Steps: {total_timesteps:,}")
    print(f"Curriculum: static → basic → medium → hard")
    print(f"Final Avg Reward: {all_rewards[-1]:.2f}")
    print(f"Final Avg Episode Length: {all_lengths[-1]:.1f}")
    print(f"Model saved to: pong_agent.zip")

    plt.switch_backend('Agg')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(all_timesteps, all_rewards, linewidth=2)
    for i in range(1, 4):
        ax1.axvline(x=i*steps_per_difficulty, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Average Reward")
    ax1.set_title("Curriculum Learning: Training Progress")
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Reward', 'Difficulty Change'])

    ax2.plot(all_timesteps, all_lengths, linewidth=2, color='orange')
    for i in range(1, 4):
        ax2.axvline(x=i*steps_per_difficulty, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Average Episode Length")
    ax2.set_title("Episode Length Over Time")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=150)
    print(f"Training metrics plot saved to training_metrics.png")
    plt.close()

    elapsed = time.time() - start
    print(f"Time taken: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print("=" * 60)

