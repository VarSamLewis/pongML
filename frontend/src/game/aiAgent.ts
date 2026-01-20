import * as tf from '@tensorflow/tfjs';
import { Ball, Paddle, GAME_CONFIG } from './types';

interface ModelWeights {
  [key: string]: number[][][] | number[][];
}

export class AIAgent {
  private model: tf.LayersModel | null = null;
  private loading = false;
  private loaded = false;

  async loadModel(): Promise<void> {
    if (this.loading || this.loaded) return;

    this.loading = true;

    try {
      console.log('[AI] Loading TensorFlow.js model...');

      // Load model weights from JSON
      const response = await fetch(`/tfjs_model/model.json?v=${Date.now()}`);
      const modelData = await response.json();

      // Build TensorFlow.js model manually from the exported weights
      this.model = this.buildModelFromWeights(modelData.weights);

      this.loaded = true;
      console.log('[AI] TensorFlow.js model loaded successfully');
    } catch (error) {
      console.error('[AI] Failed to load TensorFlow.js model:', error);
      console.log('[AI] Falling back to basic AI');
      this.loaded = false;
    } finally {
      this.loading = false;
    }
  }

  private buildModelFromWeights(weights: ModelWeights): tf.LayersModel {
    // Build the neural network architecture matching the PPO policy
    const model = tf.sequential();

    // Input layer
    model.add(tf.layers.inputLayer({ inputShape: [8] }));

    // First hidden layer (64 units, tanh activation)
    const layer1Weights = weights['policy.mlp_extractor.policy_net.0.weight'];
    const layer1Bias = weights['policy.mlp_extractor.policy_net.0.bias'];
    model.add(tf.layers.dense({
      units: 64,
      activation: 'tanh',
      weights: [
        tf.tensor2d(layer1Weights as number[][]).transpose(),
        tf.tensor1d(layer1Bias as number[])
      ]
    }));

    // Second hidden layer (64 units, tanh activation)
    const layer2Weights = weights['policy.mlp_extractor.policy_net.2.weight'];
    const layer2Bias = weights['policy.mlp_extractor.policy_net.2.bias'];
    model.add(tf.layers.dense({
      units: 64,
      activation: 'tanh',
      weights: [
        tf.tensor2d(layer2Weights as number[][]).transpose(),
        tf.tensor1d(layer2Bias as number[])
      ]
    }));

    // Output layer (3 units for 3 actions)
    const outputWeights = weights['policy.action_net.weight'];
    const outputBias = weights['policy.action_net.bias'];
    model.add(tf.layers.dense({
      units: 3,
      weights: [
        tf.tensor2d(outputWeights as number[][]).transpose(),
        tf.tensor1d(outputBias as number[])
      ]
    }));

    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

    return model;
  }

  isLoaded(): boolean {
    return this.loaded;
  }

  private createObservation(
    ball: Ball,
    aiPaddle: Paddle,
    playerPaddle: Paddle
  ): Float32Array {
    // Create 8-dimensional observation matching training environment
    // Normalized to 0-1 range
    return new Float32Array([
      ball.x / GAME_CONFIG.WIDTH,                           // ball_x
      ball.y / GAME_CONFIG.HEIGHT,                          // ball_y
      (ball.vx + 10) / 20,                                  // ball_vx (normalized)
      (ball.vy + 10) / 20,                                  // ball_vy (normalized)
      aiPaddle.y / GAME_CONFIG.HEIGHT,                      // agent_y (top edge)
      (aiPaddle.y + aiPaddle.height) / GAME_CONFIG.HEIGHT,  // agent_y + height (bottom edge)
      playerPaddle.y / GAME_CONFIG.HEIGHT,                  // opp_y (top edge)
      (playerPaddle.y + playerPaddle.height) / GAME_CONFIG.HEIGHT, // opp_y + height (bottom edge)
    ]);
  }

  async predict(
    ball: Ball,
    aiPaddle: Paddle,
    playerPaddle: Paddle
  ): Promise<'up' | 'down' | 'stay'> {
    if (!this.loaded || !this.model) {
      // Fallback to basic AI if model not loaded
      return this.basicAI(ball, aiPaddle);
    }

    try {
      const observation = this.createObservation(ball, aiPaddle, playerPaddle);

      // Run inference with TensorFlow.js
      const inputTensor = tf.tensor2d([Array.from(observation)], [1, 8]);
      const output = this.model.predict(inputTensor) as tf.Tensor;
      const actionLogits = await output.data();

      // Clean up tensors
      inputTensor.dispose();
      output.dispose();

      // Find the action with highest logit value
      let maxIdx = 0;
      let maxVal = actionLogits[0];
      for (let i = 1; i < actionLogits.length; i++) {
        if (actionLogits[i] > maxVal) {
          maxVal = actionLogits[i];
          maxIdx = i;
        }
      }

      // Map action index to movement
      // 0 = up, 1 = stay, 2 = down
      switch (maxIdx) {
        case 0:
          return 'up';
        case 1:
          return 'stay';
        case 2:
          return 'down';
        default:
          return 'stay';
      }
    } catch (error) {
      console.error('[AI] Error during inference:', error);
      return this.basicAI(ball, aiPaddle);
    }
  }

  private basicAI(ball: Ball, aiPaddle: Paddle): 'up' | 'down' | 'stay' {
    // Simple fallback AI - track ball position
    const paddleCenter = aiPaddle.y + aiPaddle.height / 2;
    const threshold = 5;

    if (paddleCenter < ball.y - threshold) {
      return 'down';
    } else if (paddleCenter > ball.y + threshold) {
      return 'up';
    } else {
      return 'stay';
    }
  }
}

// Singleton instance
export const aiAgent = new AIAgent();
