import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import time as time
import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine


def get_data():
    """Pure data extraction from DB."""
    engine = create_engine("postgresql://postgres:postgres@localhost:5432/pongML_db")
    
    query = """
        SELECT id, game_id, ball_x, ball_y, agent_y1, agent_y2, opp_y1, opp_y2 
        FROM game_data 
        WHERE winner = 'player' 
        ORDER BY game_id, id;
    """
    return pd.read_sql(query, engine)

def transform_and_normalize(df):
    """
    Convert raw database rows into normalized PPO observations and targets.
    """
    data = df.copy()

    next_y = data.groupby('game_id')['agent_y1'].shift(-1)
    
    data['target'] = 1
    data.loc[next_y < data['agent_y1'], 'target'] = 0
    data.loc[next_y > data['agent_y1'], 'target'] = 2

    data['raw_vx'] = data.groupby('game_id')['ball_x'].shift(-1) - data['ball_x']
    data['raw_vy'] = data.groupby('game_id')['ball_y'].shift(-1) - data['ball_y']

    processed = pd.DataFrame()
    processed['ball_x'] = data['ball_x'] / 800.0
    processed['ball_y'] = data['ball_y'] / 600.0
    
    processed['ball_vx'] = (data['raw_vx'] + 10.0) / 20.0
    processed['ball_vy'] = (data['raw_vy'] + 10.0) / 20.0
    
    processed['agent_y1'] = data['agent_y1'] / 600.0
    processed['agent_y2'] = (data['agent_y1'] + 25) / 600.0
    processed['opp_y1'] = data['opp_y1'] / 600.0
    processed['opp_y2'] = (data['opp_y1'] + 25) / 600.0
    
    processed['target'] = data['target']

    processed = processed.dropna()
    
    return processed


def train_model():
    model = PPO.load("pong_agent.zip")
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5) # Use a small LR to refine

    X = torch.tensor(train_df.drop('target', axis=1).values, dtype=torch.float32)
    y = torch.tensor(train_df['target'].values, dtype=torch.long)

    policy.train()
    for epoch in range(10):
        features = policy.extract_features(X)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        logits = policy.action_net(latent_pi)

        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    model.save("pong_agent_post")


if __name__ == '__main__':
    raw_df = get_data()
    train_df = transform_and_normalize(raw_df)
    train_model() 
