import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import yaml
import logging.config
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load logging configuration
with open("logging_config.yaml", "r") as logging_config_file:
    logging_config = yaml.safe_load(logging_config_file)
logging.config.dictConfig(logging_config)
logger = logging.getLogger("app_logger")

# Load model configuration from YAML file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), './config.yaml')
with open(CONFIG_PATH, "r") as config_file:
    config = yaml.safe_load(config_file)

# Extract model parameters from config and environment variables
model_name = config["model"]["name"]
max_length = config["model"].get("max_length", 100)
temperature = config["model"].get("temperature", 1.0)

# Define model and tokenizer
logger.info(f"Attempting to download model {model_name}. Please be patient...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Tokenizer downloaded successfully.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA device not available. Falling back to CPU, which may significantly impact performance.")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=os.getenv("MODEL_CACHE_DIR")).to(device)
    logger.info("Model downloaded and loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Function to perform inference with the model
def generate_text(prompt, max_length=max_length, temperature=temperature):
    logger.info("Generating text...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info(f"Generated text: {generated_text}")
    return generated_text
