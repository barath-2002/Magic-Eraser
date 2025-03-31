# Use NVIDIA CUDA Base Image (Ensures GPU support)
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install required dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    git \
    unzip \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda.sh

# Add Conda to PATH
ENV PATH="/opt/miniconda/bin:$PATH"

# Create a Conda environment with Python 3.11.11 (This is the version that supports other dependencies)
RUN conda create -n lama_env python=3.11.11 -y

# Set the working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Clone the repository
RUN git clone https://github.com/visualistapp/Magic-Eraser.git /tmp/Magic-Eraser
RUN cp -r /tmp/Magic-Eraser/* /app/ && rm -rf /tmp/Magic-Eraser

# Create a temporary directory and set TMPDIR for Pip
RUN mkdir -p /tmp/pip-tmp && chmod 777 /tmp/pip-tmp
ENV TMPDIR=/tmp/pip-tmp

# Ensure Conda environment is activated
SHELL ["/bin/bash", "-c"]
RUN echo "source activate lama_env" >> ~/.bashrc


# Install PyTorch with CUDA Support (Ensures GPU Utilization)
RUN conda install -n lama_env -y -c pytorch -c nvidia \
    pytorch torchvision torchaudio pytorch-cuda=11.8

# Install NumPy via Conda (Locks version to prevent NumPy 2.0 issues)
RUN conda install -n lama_env -y -c conda-forge \
    numpy=1.23.5

# Copy the requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN conda run -n lama_env pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8084

ENV TORCH_HOME=/app
ENV PYTHONPATH=/app
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["conda", "run", "-n", "lama_env", "uvicorn", "app:app", "--log-level", "debug","--host", "0.0.0.0", "--port", "8084"]

