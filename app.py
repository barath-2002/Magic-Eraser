from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import logging
import os
import shutil
import time
import torch
import cv2
import yaml
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from torch.utils.data._utils.collate import default_collate
from S3download import download_image

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Directories
INPUT_DIR = "/app/user_uploads/"
OUTPUT_DIR = "/app/outputs/"
MODEL_PATH = "/app/big-lama"
CHECKPOINT_PATH = f"{MODEL_PATH}/fine-tuned_lama.ckpt"

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variable for the model
loaded_model = None

def load_model():
    """Load the inpainting model only once when the app starts."""
    global loaded_model
    if loaded_model is None:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            LOGGER.info(f"Using device: {device}")

            train_config_path = os.path.join(MODEL_PATH, 'config.yaml')
            LOGGER.info(f"Loading model config from {train_config_path}")

            with open(train_config_path, 'r') as f:
                train_config = OmegaConf.create(yaml.safe_load(f))
            
            train_config.training_model.predict_only = True
            train_config.visualizer.kind = 'noop'
            LOGGER.info(f"Loading model checkpoint from {CHECKPOINT_PATH}")

            loaded_model = load_checkpoint(train_config, CHECKPOINT_PATH, strict=False, map_location=device)
            loaded_model.freeze()
            loaded_model.to(device)
            
            LOGGER.info("Model loaded successfully.")
        except Exception as e:
            LOGGER.error(f"Error loading model: {e}")
            raise e

# Initialize the model when the app starts
load_model()

app = FastAPI()

@app.post("/process/")

async def process_image(image: str = Form(...), mask: UploadFile = File(...)):
    """Processes an image and mask using the preloaded model and returns the result."""
    global loaded_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        LOGGER.info("Received image and mask for processing.")
        
        # Save uploaded image & mask
        image_path = Path(INPUT_DIR) / "image.png"
        mask_path = Path(INPUT_DIR) / "image_mask.png"
        
        image_data = download_image(image)
        with image_path.open("wb") as buffer:
            buffer.write(image_data)

        with mask_path.open("wb") as buffer:
            shutil.copyfileobj(mask.file, buffer)
        LOGGER.info(f"Saved input mask to {mask_path}")

        # Start timing
        start_time = time.time()

        # Load dataset
        LOGGER.info("Loading dataset from input directory.")
        dataset = make_default_val_dataset(indir=INPUT_DIR, kind="default", img_suffix=".png", pad_out_to_modulo=8)


        if len(dataset) == 0:
            LOGGER.error("Dataset is empty. Ensure that the input image and mask are in the correct format.")
            return {"error": "Dataset is empty. Ensure valid input images and masks."}

        LOGGER.info("Collating batch for model inference.")
        batch = move_to_device(default_collate([dataset[0]]), device)
        
        if 'mask' not in batch:
            LOGGER.error("Mask key is missing in the batch.")
            return {"error": "Mask not found in the dataset."}

        batch['mask'] = ((batch['mask'] > 0)).float().to(device)
        LOGGER.info(f"Batch loaded. Keys available: {batch.keys()}")

        # Perform inpainting
        LOGGER.info("Running model inference.")
        batch = loaded_model(batch)

        if 'inpainted' not in batch:
            LOGGER.error("Key 'inpainted' is missing in model output.")
            return {"error": "Model did not return 'inpainted' result."}

        cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
        LOGGER.info(f"Inpainting completed. Output shape: {cur_res.shape}")

        # Post-processing
        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        output_image_path = Path(OUTPUT_DIR) / "output.png"
        cv2.imwrite(str(output_image_path), cur_res)
        LOGGER.info(f"Saved output image to {output_image_path}")

        # End timing
        processing_time = time.time() - start_time
        LOGGER.info(f"Processing completed in {processing_time:.2f} seconds")

        # ✅ **Free GPU Memory**
        del batch  # Delete the batch tensor
        torch.cuda.empty_cache()  # Clear unused memory

        if output_image_path.exists():
            response = FileResponse(output_image_path, media_type="image/png")
            response.headers["X-Processing-Time"] = f"{processing_time:.2f} seconds"
            return response

        LOGGER.error("Output file does not exist after processing.")
        return {"error": "Inpainting failed", "processing_time": f"{processing_time:.2f} seconds"}

    except Exception as e:
        LOGGER.error(f"Error during processing: {e}")
        return {"error": f"Unexpected error: {str(e)}"}
