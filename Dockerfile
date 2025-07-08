# Start from the official Miniconda image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Install system dependencies needed by pyvips and OpenCV
RUN apt-get update && \
    apt-get install -y libgl1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy your local files into the container
COPY . .

# Create Conda environment (non-interactive)
RUN conda create --name pebsi-detection python=3.11 -y

# Download flash-attn wheel compatible with Python 3.11 and CUDA 11
RUN wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl


# Install Python packages using pip inside the Conda env
RUN conda run -n pebsi-detection pip install --no-cache-dir -r requirements.txt

# Install the local package in editable mode
RUN conda run -n pebsi-detection pip install -e .

# Cleanup the wheel file to save space
RUN rm flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Ensure future commands use the conda env
SHELL ["conda", "run", "-n", "pebsi-detection", "/bin/bash", "-c"]

# Expose FastAPI port
EXPOSE 8001

# Start the FastAPI app using Uvicorn from within the Conda env
CMD ["conda", "run", "--no-capture-output", "-n", "pebsi-detection", "python", "main.py"]
