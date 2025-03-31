# Magic-Eraser
# FastAPI Inpainting Service  

This FastAPI application provides an endpoint for image inpainting using a preloaded deep learning model.  

## Key Components  

### 1. **Model Loading**  
- Loads the LaMa inpainting model (`big-lama`) on startup.  
- Uses GPU if available; otherwise, defaults to CPU.  
- Reads configuration from `config.yaml` and loads the model checkpoint.  

### 2. **FastAPI Endpoint (`/process/`)**  
- Accepts an **image** and **mask** as input files.  
- Saves the uploaded files to a designated directory.  
- Loads the dataset and prepares the batch for inference.  
- Runs the inpainting model and processes the output.  
- Saves the inpainted image and returns it as a response.  

### 3. **Logging & Error Handling**  
- Logs important steps for debugging.  
- Handles missing inputs, empty datasets, and model inference errors.  

### 4. **Performance Optimization**  
- Uses `torch.cuda.empty_cache()` to free GPU memory after inference.  
- Measures and returns processing time in headers (`X-Processing-Time`).  

## Usage  
- Start the API server.  
- Send a POST request to `/process/` with an image and mask.  
- Receive the inpainted image as a response.  
