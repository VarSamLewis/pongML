import { useState, useEffect, useRef, useCallback } from 'react';
import { GameCanvas } from './GameCanvas';
import { GameState, FrameData, GAME_CONFIG } from '../game/types';
import {
  createInitialBall,
  createInitialPaddle,
  updateBallPosition,
  checkWallCollision,
  checkPaddleCollision,
  checkScore,
  movePaddle,
} from '../game/physics';
import { aiAgent } from '../game/aiAgent';
import { submitGameData } from '../utils/api';

export function PongGame() {
  const [gameState, setGameState] = useState<GameState>({
    ball: createInitialBall(),
    playerPaddle: createInitialPaddle(),
    aiPaddle: createInitialPaddle(),
    playerScore: 0,
    aiScore: 0,
    status: 'menu',
    winner: null,
  });

  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelError, setModelError] = useState<string>('');
  const [submissionStatus, setSubmissionStatus] = useState<string>('');

  const frameDataRef = useRef<FrameData[]>([]);
  const currentGameIdRef = useRef<string>('');
  const keysPressed = useRef<Set<string>>(new Set());
  const animationFrameId = useRef<number>(0);

  // Load ONNX model on mount
  useEffect(() => {
    aiAgent.loadModel().then(() => {
      if (aiAgent.isLoaded()) {
        setModelLoaded(true);
      } else {
        setModelError('Failed to load AI model. Please check console for errors.');
      }
    }).catch((error) => {
      setModelError(`Error loading AI model: ${error.message}`);
    });
  }, []);

  // Collect frame data (without winner - we'll add it later when game ends)
  const collectFrameData = useCallback((state: GameState) => {
    const frameData: FrameData = {
      game_id: currentGameIdRef.current,
      winner: 'player', // Placeholder - will be updated when game ends
      ball_x: state.ball.x / GAME_CONFIG.WIDTH,
      ball_y: state.ball.y / GAME_CONFIG.HEIGHT,
      agent_y1: state.aiPaddle.y / GAME_CONFIG.HEIGHT,
      agent_y2: (state.aiPaddle.y + state.aiPaddle.height) / GAME_CONFIG.HEIGHT,
      opp_y1: state.playerPaddle.y / GAME_CONFIG.HEIGHT,
      opp_y2: (state.playerPaddle.y + state.playerPaddle.height) / GAME_CONFIG.HEIGHT,
    };
    frameDataRef.current.push(frameData);
  }, []);

  // Submit game data to backend
  const submitData = useCallback(async (winner: 'player' | 'ai') => {
    if (frameDataRef.current.length === 0) return;

    // Update all frames with the actual winner
    frameDataRef.current.forEach(frame => {
      frame.winner = winner;
    });

    console.log(`[DATA] Submitting ${frameDataRef.current.length} frames for game ${currentGameIdRef.current}`);
    console.log(`[DATA] Winner: ${winner}`);

    setSubmissionStatus('Submitting game data...');
    try {
      const today = new Date().toISOString().split('T')[0];
      await submitGameData(frameDataRef.current, today);
      console.log(`[DATA] Successfully submitted ${frameDataRef.current.length} frames`);
      setSubmissionStatus(`Successfully submitted ${frameDataRef.current.length} frames!`);
      setTimeout(() => setSubmissionStatus(''), 3000);
    } catch (error) {
      console.error('[DATA] Failed to submit game data:', error);
      setSubmissionStatus('Failed to submit data. Check console for errors.');
      setTimeout(() => setSubmissionStatus(''), 3000);
    }
    frameDataRef.current = [];
  }, []);

  // Game loop
  useEffect(() => {
    if (gameState.status !== 'playing') {
      return;
    }

    const gameLoop = async () => {
      setGameState((prevState) => {
        const newState = { ...prevState };

        // Handle player input
        let playerDirection: 'up' | 'down' | 'stay' = 'stay';
        if (keysPressed.current.has('ArrowUp')) {
          playerDirection = 'up';
        } else if (keysPressed.current.has('ArrowDown')) {
          playerDirection = 'down';
        }
        movePaddle(newState.playerPaddle, playerDirection);

        // AI will be updated asynchronously
        return newState;
      });

      // Run AI inference (async)
      let aiDirection: 'up' | 'down' | 'stay' = 'stay';
      try {
        aiDirection = await aiAgent.predict(
          gameState.ball,
          gameState.aiPaddle,
          gameState.playerPaddle
        );
      } catch (error) {
        console.error('[AI] Prediction failed:', error);
        // Stop the game if model fails
        setGameState((prev) => ({
          ...prev,
          status: 'gameover',
          winner: null,
        }));
        return;
      }

      setGameState((prevState) => {
        const newState = { ...prevState };

        // Move AI paddle
        movePaddle(newState.aiPaddle, aiDirection);

        // Update ball position
        updateBallPosition(newState.ball);

        // Check wall collisions
        checkWallCollision(newState.ball);

        // Check paddle collisions
        checkPaddleCollision(newState.ball, newState.aiPaddle, true); // AI paddle on left
        checkPaddleCollision(newState.ball, newState.playerPaddle, false); // Player paddle on right

        // Check scoring
        const scored = checkScore(newState.ball);
        if (scored === 'left') {
          // Ball went past AI paddle (left) - Player scores
          newState.playerScore += 1;
          newState.status = 'gameover';
          newState.winner = 'player';
          console.log('[GAME] Game ended - PLAYER WINS!');
        } else if (scored === 'right') {
          // Ball went past Player paddle (right) - AI scores
          newState.aiScore += 1;
          newState.status = 'gameover';
          newState.winner = 'ai';
          console.log('[GAME] Game ended - AI WINS!');
        }

        // Collect frame data
        collectFrameData(newState);

        return newState;
      });

      animationFrameId.current = requestAnimationFrame(gameLoop);
    };

    animationFrameId.current = requestAnimationFrame(gameLoop);

    return () => {
      cancelAnimationFrame(animationFrameId.current);
    };
  }, [gameState.status, collectFrameData]);

  // Handle game over - submit data
  useEffect(() => {
    if (gameState.status === 'gameover' && gameState.winner) {
      submitData(gameState.winner);
    }
  }, [gameState.status, gameState.winner, submitData]);

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        e.preventDefault();
        keysPressed.current.add(e.key);
      }

      if (e.key === 'Enter') {
        e.preventDefault();
        setGameState((prev) => {
          if (prev.status === 'menu' || prev.status === 'gameover') {
            // Start new game - generate new UUID
            frameDataRef.current = [];
            const newGameId = crypto.randomUUID();
            currentGameIdRef.current = newGameId;
            console.log(`[GAME] Starting new game with ID: ${newGameId}`);
            return {
              ball: createInitialBall(),
              playerPaddle: createInitialPaddle(),
              aiPaddle: createInitialPaddle(),
              playerScore: 0,
              aiScore: 0,
              status: 'playing',
              winner: null,
            };
          } else if (prev.status === 'playing') {
            // Pause game
            console.log('[GAME] Game paused');
            return { ...prev, status: 'paused' };
          } else if (prev.status === 'paused') {
            // Resume game
            console.log('[GAME] Game resumed');
            return { ...prev, status: 'playing' };
          }
          return prev;
        });
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        keysPressed.current.delete(e.key);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  return (
    <div className="flex flex-col items-center gap-6 p-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-[#fbf0df] mb-2">Pong ML</h1>
        <p className="text-[#f3d5a3] text-sm">
          {modelError ? `AI Model Error: ${modelError}` : modelLoaded ? 'AI Model Loaded ✓' : 'Loading AI Model...'}
        </p>
      </div>

      <GameCanvas gameState={gameState} />

      {modelError && (
        <div className="bg-[#1a1a1a] border-2 border-red-500 rounded-lg px-4 py-2 text-red-400 font-mono text-sm">
          {modelError}
        </div>
      )}

      {submissionStatus && (
        <div className="bg-[#1a1a1a] border-2 border-[#f3d5a3] rounded-lg px-4 py-2 text-[#fbf0df] font-mono text-sm">
          {submissionStatus}
        </div>
      )}

      <div className="bg-[#1a1a1a] border-2 border-[#fbf0df] rounded-lg p-4 max-w-md text-[#fbf0df] font-mono text-sm">
        <h2 className="font-bold text-[#f3d5a3] mb-2">Controls</h2>
        <ul className="space-y-1">
          <li>↑ Arrow Up - Move paddle up</li>
          <li>↓ Arrow Down - Move paddle down</li>
          <li>⏎ Enter - Start / Pause / Restart</li>
        </ul>
        <p className="mt-3 text-xs text-[#fbf0df80]">
          Game data is collected during play and submitted to the backend after each game for continuous training.
        </p>
      </div>
    </div>
  );
}
