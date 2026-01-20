# Pong ML Frontend

A browser-based Pong game where you play against an AI agent trained with reinforcement learning. Game data is collected and sent to the backend for continuous training.

## Architecture

### Tech Stack
- **Runtime**: Bun
- **Framework**: React 19 + TypeScript
- **Styling**: Tailwind CSS
- **AI Inference**: onnxruntime-web (browser-based ONNX model execution)
- **Backend Communication**: REST API

### Project Structure
```
frontend/
├── public/
│   └── models/
│       └── pong_agent.onnx        # AI agent model
├── src/
│   ├── components/
│   │   ├── GameCanvas.tsx         # Canvas rendering
│   │   └── PongGame.tsx           # Game logic & loop
│   ├── game/
│   │   ├── types.ts               # TypeScript interfaces
│   │   ├── physics.ts             # Ball/paddle physics
│   │   └── aiAgent.ts             # ONNX model loader & inference
│   ├── utils/
│   │   └── api.ts                 # Backend API calls
│   ├── App.tsx                    # Main app component
│   ├── index.ts                   # Bun server
│   └── index.html                 # Entry HTML
├── .env                           # Environment variables
└── package.json
```

## Game Mechanics

### Dimensions
- Canvas: 800×600 pixels
- Paddle: 10px × 25px
- Ball: 8px radius
- Paddle speed: 5px/frame
- Ball speed: 5px/frame (increases 5% on each hit)

### Controls
- **↑ Arrow Up**: Move paddle up
- **↓ Arrow Down**: Move paddle down
- **⏎ Enter**: Start / Pause / Restart game

### Gameplay
- Player controls the **right paddle**
- AI controls the **left paddle**
- Game ends when either side scores (first point wins)
- Frame data is collected during gameplay
- Data is automatically submitted to backend after game ends

## Data Collection

### Frame Data Format
Each frame during gameplay collects:
```typescript
{
  ball_x: number,      // Normalized 0-1
  ball_y: number,      // Normalized 0-1
  agent_y1: number,    // AI paddle top edge (normalized)
  agent_y2: number,    // AI paddle bottom edge (normalized)
  opp_y1: number,      // Player paddle top edge (normalized)
  opp_y2: number       // Player paddle bottom edge (normalized)
}
```

### Submission
- Data is batched and sent after each game
- Endpoint: `POST /api/game-data`
- Payload: `{ frames: FrameData[], date: string }`

## AI Agent

### ONNX Model
- **Input**: 8-dimensional observation vector
  - ball_x, ball_y (normalized)
  - ball_vx, ball_vy (normalized)
  - agent_y, agent_y + paddle_height (normalized)
  - opp_y, opp_y + paddle_height (normalized)

- **Output**: Action (0 = up, 1 = stay, 2 = down)

### Model Updates
When you retrain the model:
1. Export new ONNX file: `python r_training/export_to_onnx.py`
2. Copy to frontend: `cp r_training/pong_agent.onnx frontend/public/models/`
3. Model is automatically cache-busted with timestamp query param

### Fallback AI
If ONNX model fails to load, a simple heuristic AI activates:
- Tracks ball Y position
- Moves paddle to intercept

## Development

### Run Dev Server
```bash
cd frontend
bun --hot src/index.ts
```

Frontend runs on `http://localhost:3000` (default Bun port)

### Environment Variables
Edit `.env` to configure backend URL:
```bash
BACKEND_URL=http://localhost:3001
```

### Build for Production
```bash
bun run build
```

## Backend Integration

The frontend expects a backend server with:
- **Endpoint**: `POST /api/game-data`
- **Request Body**: `{ frames: FrameData[], date: string }`
- **Response**: `{ success: boolean, message: string }`

Currently, the frontend's `/api/game-data` route is a placeholder that logs data. You'll implement the actual PostgreSQL insertion in the separate backend server.

## Notes

- Game runs at 60 FPS
- Canvas scales responsively on smaller screens
- Modern minimal design with Tailwind theme colors
- No authentication (as requested)
- CORS will need to be handled if frontend and backend are on different origins
