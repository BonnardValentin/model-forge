from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_gpt_neo import generate_text 
import os
from dotenv import load_dotenv
import logging.config
import yaml
import torch

# Load environment variables
load_dotenv()

# Load logging configuration
with open("logging_config.yaml", "r") as logging_config_file:
    logging_config = yaml.safe_load(logging_config_file)
logging.config.dictConfig(logging_config)
logger = logging.getLogger("app_logger")

# Create FastAPI app
app = FastAPI()

# Define request body format
class PredictRequest(BaseModel):
    prompt: str
    max_length: int = 150
    temperature: float = 0.9

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        logger.info(f"Received prediction request with prompt: {request.prompt}")
        result = generate_text(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        logger.info(f"Generated text: {result}")
        return {"generated_text": result}
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA out of memory error: {str(e)}")
        raise HTTPException(status_code=500, detail="CUDA out of memory. Try reducing the max_length or temperature.")
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")
@app.get("/health")
async def health():
    try:
        # This is a simple check to confirm the app is running
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "details": str(e)}