# Use an official NVIDIA CUDA image with Python 3.8
FROM python:3.10.17-bullseye

# Set environment variables to avoid interactive prompts and upgrade pip
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y
RUN apt-get update && apt-get install libgl1 -y
RUN apt-get install pip sudo software-properties-common git curl -y

# Set working directory
WORKDIR /app

# Instal MMDetections
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -U openmim==0.3.9
RUN mim install mmengine==0.10.4
RUN mim install mmcv==2.1.0
RUN mim install mmdet==3.3.0
RUN mim install mmpretrain==1.2.0

# Copy requirements file and install dependencies
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "growth_monitoring_pipeline:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--timeout-keep-alive", "5"]
