import { Ball, Paddle, GAME_CONFIG } from './types';

export function updateBallPosition(ball: Ball): void {
  ball.x += ball.vx;
  ball.y += ball.vy;
}

export function checkWallCollision(ball: Ball): void {
  // Top wall collision
  if (ball.y <= ball.radius) {
    ball.vy *= -1;
    ball.y = ball.radius;
  }

  // Bottom wall collision
  if (ball.y >= GAME_CONFIG.HEIGHT - ball.radius) {
    ball.vy *= -1;
    ball.y = GAME_CONFIG.HEIGHT - ball.radius;
  }
}

export function checkPaddleCollision(
  ball: Ball,
  paddle: Paddle,
  isLeftPaddle: boolean
): boolean {
  const paddleX = isLeftPaddle ? GAME_CONFIG.PADDLE_WIDTH : GAME_CONFIG.WIDTH - GAME_CONFIG.PADDLE_WIDTH;
  const ballEdge = isLeftPaddle ? ball.x - ball.radius : ball.x + ball.radius;

  // Check if ball is at paddle X position
  const atPaddleX = isLeftPaddle
    ? ballEdge <= paddleX
    : ballEdge >= paddleX;

  if (atPaddleX) {
    // Check if ball is within paddle Y range
    if (ball.y >= paddle.y && ball.y <= paddle.y + paddle.height) {
      // Collision detected - reverse X velocity
      ball.vx = isLeftPaddle ? Math.abs(ball.vx) : -Math.abs(ball.vx);

      // Reposition ball to prevent stuck-in-paddle bug
      ball.x = isLeftPaddle
        ? paddleX + ball.radius
        : paddleX - ball.radius;

      // Add spin based on where ball hit paddle (like training env)
      const hitPos = (ball.y - paddle.y) / paddle.height - 0.5;
      ball.vy += hitPos * 3;

      // Increase ball speed by 5%
      ball.vx *= GAME_CONFIG.BALL_SPEED_INCREASE;
      ball.vy *= GAME_CONFIG.BALL_SPEED_INCREASE;

      return true;
    }
  }

  return false;
}

export function checkScore(ball: Ball): 'left' | 'right' | null {
  // Ball went past left paddle (AI scored)
  if (ball.x <= 0) {
    return 'left';
  }

  // Ball went past right paddle (Player scored)
  if (ball.x >= GAME_CONFIG.WIDTH) {
    return 'right';
  }

  return null;
}

export function movePaddle(paddle: Paddle, direction: 'up' | 'down' | 'stay'): void {
  if (direction === 'up') {
    paddle.y = Math.max(0, paddle.y - paddle.speed);
  } else if (direction === 'down') {
    paddle.y = Math.min(
      GAME_CONFIG.HEIGHT - paddle.height,
      paddle.y + paddle.speed
    );
  }
}

export function createInitialBall(): Ball {
  // Random angle between -45 and 45 degrees
  const angle = (Math.random() * Math.PI / 2) - Math.PI / 4;

  // Random direction (left or right)
  const direction = Math.random() < 0.5 ? -1 : 1;

  return {
    x: GAME_CONFIG.WIDTH / 2,
    y: GAME_CONFIG.HEIGHT / 2,
    vx: direction * GAME_CONFIG.BALL_SPEED_INITIAL * Math.cos(angle),
    vy: GAME_CONFIG.BALL_SPEED_INITIAL * Math.sin(angle),
    radius: GAME_CONFIG.BALL_RADIUS,
  };
}

export function createInitialPaddle(): Paddle {
  return {
    y: GAME_CONFIG.HEIGHT / 2 - GAME_CONFIG.PADDLE_HEIGHT / 2,
    width: GAME_CONFIG.PADDLE_WIDTH,
    height: GAME_CONFIG.PADDLE_HEIGHT,
    speed: GAME_CONFIG.PADDLE_SPEED,
  };
}
