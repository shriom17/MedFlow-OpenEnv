# Main entry point for MedFlow OpenEnv
from fastapi import FastAPI
from app.tasks import get_tasks
from app.env   import MedFlowEnv

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to MedFlow OpenEnv"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/tasks")
async def tasks():
    return get_tasks()