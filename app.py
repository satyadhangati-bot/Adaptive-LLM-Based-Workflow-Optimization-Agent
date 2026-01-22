from fastapi import FastAPI
from src.trainer import train

app = FastAPI(title="Adaptive Workflow Optimization Agent")

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/train")
def train_agent():
    train(episodes=20)
    return {"status": "training completed"}
