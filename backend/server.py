from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from pydantic import BaseModel
from typing import List 
import logging  
import uuid 

logger = logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"], 
    allow_headers=["*"],
)

class FrameData(BaseModel):
    game_id: uuid.UUID,
    winner: str
    ball_x: float
    ball_y: float
    agent_y1: float
    agent_y2: float 
    opp_y1: float 
    opp_y2: float 

class GameDataSubmission(BaseModel):
    frames: List[FrameData]
    date: str

@app.post("/api/game-data")
async def recieve_game_data(data: GameDataSubmission):
    conn = psycopg2.connect(
        dbname="pongML_db",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    for frame in data.frames:
        try:
            cursor.execute(""" 
                INSERT INTO game_data (date, game_id, winner ,ball_x, ball_y, agent_y1, agent_y2, opp_y1, opp_y2)
                VALUES (%s, $s, %s, ,%s, %s, %s, %s, %s, %s)
            """, (data.date, frame.game_id, frame.winner, frame.ball_x, frame.ball_y, frame.agent_y1, frame.agent_y2, frame.opp_y1, frame.opp_y2))
        except Exception as e:
            logger.error("Frame not written")

    logger.info(f"{frame.ame_id} Frames inserted into DB")
    conn.commit()
    cursor.close()
    conn.close()

    return {"success": True, "message": f"Inserted {len(data.frames)} frames"}
    
