import { FrameData, GameDataSubmission } from '../game/types';

// Backend URL - change this if your backend runs on a different port
const BACKEND_URL = 'http://localhost:3001';

export async function submitGameData(
  frames: FrameData[],
  date: string
): Promise<void> {
  const data: GameDataSubmission = {
    frames,
    date,
  };

  const response = await fetch(`${BACKEND_URL}/api/game-data`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to submit game data: ${response.status} ${errorText}`);
  }

  const result = await response.json();
  console.log('Game data submitted successfully:', result);
}
