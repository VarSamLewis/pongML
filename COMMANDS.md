# Pong ML - Setup and Running Commands

This document explains all the commands needed to start and run the Pong ML application.

## Prerequisites

- PostgreSQL installed and running
- Python 3.12+ with venv
- Bun runtime installed
- Node.js (for dependencies)

## 1. PostgreSQL Database

### Start PostgreSQL Service

```bash
# On Linux (systemd)
sudo systemctl start postgresql

# On macOS (Homebrew)
brew services start postgresql

# On Windows
# Use pgAdmin or Windows Services to start PostgreSQL
```

### Verify PostgreSQL is Running

```bash
sudo systemctl status postgresql
# or
pg_isready
```

### Create Database (First Time Only)

```bash
# Connect to PostgreSQL as postgres user
sudo -u postgres psql

# Inside psql, create the database
CREATE DATABASE pongML_db;

# Create the game_data table
\c pongML_db

CREATE TABLE game_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    game_id UUID NOT NULL,
    winner VARCHAR(10) NOT NULL,
    ball_x FLOAT NOT NULL,
    ball_y FLOAT NOT NULL,
    agent_y1 FLOAT NOT NULL,
    agent_y2 FLOAT NOT NULL,
    opp_y1 FLOAT NOT NULL,
    opp_y2 FLOAT NOT NULL
);

# Exit psql
\q
```

## 2. Backend (FastAPI Server)

### Navigate to Backend Directory

```bash
cd backend
```

### Activate Virtual Environment (if not already activated)

```bash
source .venv/bin/activate
```

### Install Dependencies (First Time Only)

```bash
pip install fastapi uvicorn psycopg2-binary
```

### Start Backend Server

```bash
uvicorn server:app --port 3001
```

The backend will run on **http://localhost:3001**

### Backend Environment

- **Database**: pongML_db
- **User**: postgres
- **Password**: postgres
- **Host**: localhost
- **Port**: 5432

## 3. Frontend (Bun + React)

### Navigate to Frontend Directory

```bash
cd frontend
```

### Install Dependencies (First Time Only)

```bash
bun install
```

### Start Frontend Development Server

```bash
bun run --hot src/index.ts
```

The frontend will run on **http://localhost:3000**

## Quick Start - All Services

Open 3 terminal windows and run:

### Terminal 1: PostgreSQL
```bash
sudo systemctl start postgresql
# Verify it's running
pg_isready
```

### Terminal 2: Backend
```bash
cd backend
source .venv/bin/activate
uvicorn server:app --port 3001
```

### Terminal 3: Frontend
```bash
cd frontend
bun run --hot src/index.ts
```

## Accessing the Application

1. Open your browser to **http://localhost:3000**
2. Press **Enter** to start a game
3. Use **↑** and **↓** arrow keys to move your paddle
4. Game data is automatically submitted to the backend after each game

## Troubleshooting

### PostgreSQL Connection Error

If you see `could not connect to server`:
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Start it if it's not running
sudo systemctl start postgresql
```

### Backend Port Already in Use

If port 3001 is already in use:
```bash
# Find the process using the port
lsof -i :3001

# Kill the process
kill -9 <PID>

# Or use a different port
uvicorn server:app --port 3002
# (Don't forget to update frontend/src/utils/api.ts)
```

### Frontend Port Already in Use

If port 3000 is already in use:
```bash
# Kill the process on port 3000
lsof -i :3000
kill -9 <PID>
```

### Model Not Loading

If the AI model fails to load:
```bash
# Check if the model file exists
ls -lh frontend/public/tfjs_model/model.json

# If missing, regenerate it from the training directory
cd r_training
python export_to_tfjs.py
```

## Stopping Services

### Stop Frontend
Press **Ctrl+C** in the frontend terminal

### Stop Backend
Press **Ctrl+C** in the backend terminal

### Stop PostgreSQL (Optional)
```bash
sudo systemctl stop postgresql
```

## Database Queries

### View Game Data

```bash
# Connect to the database
sudo -u postgres psql -d pongML_db

# View all games
SELECT DISTINCT game_id, winner, COUNT(*) as frame_count
FROM game_data
GROUP BY game_id, winner
ORDER BY MAX(id) DESC
LIMIT 10;

# View total frame count
SELECT COUNT(*) FROM game_data;

# Exit
\q
```
