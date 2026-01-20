import { useEffect, useRef } from 'react';
import { GameState, GAME_CONFIG } from '../game/types';

interface GameCanvasProps {
  gameState: GameState;
}

export function GameCanvas({ gameState }: GameCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, GAME_CONFIG.WIDTH, GAME_CONFIG.HEIGHT);

    // Draw center line
    ctx.strokeStyle = '#fbf0df40';
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 10]);
    ctx.beginPath();
    ctx.moveTo(GAME_CONFIG.WIDTH / 2, 0);
    ctx.lineTo(GAME_CONFIG.WIDTH / 2, GAME_CONFIG.HEIGHT);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw AI paddle (left side)
    ctx.fillStyle = '#f3d5a3';
    ctx.shadowBlur = 10;
    ctx.shadowColor = '#f3d5a3';
    ctx.fillRect(
      0,
      gameState.aiPaddle.y,
      gameState.aiPaddle.width,
      gameState.aiPaddle.height
    );

    // Draw player paddle (right side)
    ctx.fillStyle = '#fbf0df';
    ctx.shadowBlur = 10;
    ctx.shadowColor = '#fbf0df';
    ctx.fillRect(
      GAME_CONFIG.WIDTH - gameState.playerPaddle.width,
      gameState.playerPaddle.y,
      gameState.playerPaddle.width,
      gameState.playerPaddle.height
    );

    // Draw ball
    ctx.fillStyle = '#ffffff';
    ctx.shadowBlur = 15;
    ctx.shadowColor = '#ffffff';
    ctx.beginPath();
    ctx.arc(
      gameState.ball.x,
      gameState.ball.y,
      gameState.ball.radius,
      0,
      Math.PI * 2
    );
    ctx.fill();

    // Reset shadow for text
    ctx.shadowBlur = 0;

    // Draw scores
    ctx.fillStyle = '#fbf0df';
    ctx.font = 'bold 48px monospace';
    ctx.textAlign = 'center';

    // AI score (left)
    ctx.fillText(
      gameState.aiScore.toString(),
      GAME_CONFIG.WIDTH / 4,
      60
    );

    // Player score (right)
    ctx.fillText(
      gameState.playerScore.toString(),
      (GAME_CONFIG.WIDTH * 3) / 4,
      60
    );

    // Draw labels
    ctx.font = 'bold 14px monospace';
    ctx.fillStyle = '#fbf0df80';
    ctx.fillText('AI', GAME_CONFIG.WIDTH / 4, 90);
    ctx.fillText('PLAYER', (GAME_CONFIG.WIDTH * 3) / 4, 90);

    // Draw status overlays
    if (gameState.status === 'menu') {
      drawOverlay(ctx, 'PONG ML', 'Press ENTER to start');
    } else if (gameState.status === 'paused') {
      drawOverlay(ctx, 'PAUSED', 'Press ENTER to resume');
    } else if (gameState.status === 'gameover') {
      const winnerText = gameState.winner === 'player' ? 'PLAYER WINS!' : 'AI WINS!';
      drawOverlay(ctx, winnerText, 'Press ENTER to restart');
    }
  }, [gameState]);

  function drawOverlay(
    ctx: CanvasRenderingContext2D,
    title: string,
    subtitle: string
  ) {
    // Semi-transparent overlay
    ctx.fillStyle = '#1a1a1a99';
    ctx.fillRect(0, 0, GAME_CONFIG.WIDTH, GAME_CONFIG.HEIGHT);

    // Title
    ctx.fillStyle = '#fbf0df';
    ctx.font = 'bold 64px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(title, GAME_CONFIG.WIDTH / 2, GAME_CONFIG.HEIGHT / 2 - 20);

    // Subtitle
    ctx.font = 'bold 20px monospace';
    ctx.fillStyle = '#f3d5a3';
    ctx.fillText(subtitle, GAME_CONFIG.WIDTH / 2, GAME_CONFIG.HEIGHT / 2 + 40);

    // Instructions
    ctx.font = '16px monospace';
    ctx.fillStyle = '#fbf0df80';
    ctx.fillText('Use ↑↓ arrows to move', GAME_CONFIG.WIDTH / 2, GAME_CONFIG.HEIGHT - 80);
  }

  return (
    <canvas
      ref={canvasRef}
      width={GAME_CONFIG.WIDTH}
      height={GAME_CONFIG.HEIGHT}
      className="border-2 border-[#fbf0df] rounded-lg shadow-2xl"
      style={{
        maxWidth: '100%',
        height: 'auto',
        imageRendering: 'pixelated',
      }}
    />
  );
}
