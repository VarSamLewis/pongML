export interface Ball {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
}

export interface Paddle {
  y: number;
  width: number;
  height: number;
  speed: number;
}

export interface GameState {
  ball: Ball;
  playerPaddle: Paddle;
  aiPaddle: Paddle;
  playerScore: number;
  aiScore: number;
  status: 'menu' | 'playing' | 'paused' | 'gameover';
  winner: 'player' | 'ai' | null;
}

export interface FrameData {
  game_id: string;
  winner: 'player' | 'ai';
  ball_x: number;
  ball_y: number;
  agent_y1: number;
  agent_y2: number;
  opp_y1: number;
  opp_y2: number;
}

export interface GameDataSubmission {
  frames: FrameData[];
  date: string;
}

export const GAME_CONFIG = {
  WIDTH: 1200,
  HEIGHT: 800,
  PADDLE_WIDTH: 10,
  PADDLE_HEIGHT: 50,
  BALL_RADIUS: 8,
  PADDLE_SPEED: 5,
  BALL_SPEED_INITIAL: 5,
  BALL_SPEED_INCREASE: 1.05,
  FPS: 60,
} as const;
