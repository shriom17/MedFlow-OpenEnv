# Main entry point for MedFlow OpenEnv
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to MedFlow OpenEnv"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
