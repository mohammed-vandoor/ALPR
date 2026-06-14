FROM python:3.11-slim

# System deps for OpenCV + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install PyTorch CPU (smaller, no CUDA needed for HF free tier)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    streamlit opencv-python-headless fast-alpr Pillow numpy \
    onnxruntime ultralytics timm huggingface_hub \
    fastapi "uvicorn[standard]" python-multipart \
    websockets pandas pyarrow

# Install transformers separately with force-reinstall to avoid stale cache
RUN pip install --no-cache-dir --force-reinstall "transformers==4.41.2" tokenizers
RUN python -c "import transformers; print('transformers version:', transformers.__version__)"

# Copy app code
COPY . .

# HF Spaces exposes port 7860
EXPOSE 7860

# Start FastAPI server on port 7860
CMD ["python", "server.py", "--port", "7860"]
