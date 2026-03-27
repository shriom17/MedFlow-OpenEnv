# Main entry point for MedFlow OpenEnv
from argparse import Action

from fastapi import FastAPI
from app.tasks import get_tasks
from app.env   import MedFlowEnv
from app.grader import Grader
from app.baseline import run_baseline

app = FastAPI()
env = MedFlowEnv()
grader = Grader()
@app.get("/")
async def root():
    return {"message": "Welcome to MedFlow OpenEnv"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/tasks")
async def tasks():
    return get_tasks()

@app.get("/grader")
def get_score():
    return {"score": grader.get_score()}

@app.get("/baseline")
def baseline():
    return {"baseline_score": get_score}

@app.get("/reset")
def reset():
    global grader
    grader = Grader()   
    return env.reset()

@app.post("/step")
def step(action: Action):
    state, reward, done = env.step(action)

    grader.add_reward(reward)  

    return {
        "state": state,
        "reward": reward,
        "done": done
    }