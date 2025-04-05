# Growth tracking instance segmentation pipeline for oyster mushroom clusters


This repository contains an instance segmentation pipeline implementation for oyster mushroom clusters growth monitoring. Docker is used to create an image with CUDA 11.8 with Jupyter Lab for editing the scripts interactively. **Currently, this repository is not a complete tool for direct integration into other projects, but acts as a code baseline for future implementations.**

## Getting Started

In order to run the Docker image, you must first have NVIDIA Container Toolkit installed on the host machine  (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Prerequisites

- Docker
- Python 3.8+
- NVIDIA Container Toolkit

### Installation:

1. **Clone the repository:**

```bash
git clone https://github.com/Chris-Charisis/Oyster-Mushrooms-Growth-Tracking-Instance-Segmentation-Pipeline.git
cd Oyster-Mushrooms-Growth-Tracking-Instance-Segmentation-Pipeline
```

2. **Build the Docker image:**

```bash
docker build -t growth-monitoring-pipeline-image .
```

3. **Run the Docker container:**

```bash
docker run -d --name growth-monitoring-pipeline --gpus all -p 8889:8888 -v $(pwd):/app -it growth-monitoring-pipeline-image
```

4. **Open browser tab to access Jupyter:**
   http://localhost:8889/lab

   

5. (Optional) Stop and remove the container after use:
```bash
docker stop growth-monitoring-pipeline
docker rm growth-monitoring-pipeline
```

## Running the Python Script (Non-interactive Mode)
In addition to using the Jupyter Notebook, the pipeline functionality can also be executed directly using the Python script version.

### Steps:
1. Make sure the container is installed and running. If installed but not running, start the container:
```bash
docker start growth-monitoring-pipeline
```
2. Execute the Python script inside the container:
```bash
docker exec -it growth-monitoring-pipeline python3 growth_monitoring_pipeline.py
```
This will run the script and generate the results in the results/ directory as described below.
3. (Optional) Stop and remove the container after use:
```bash
docker stop growth-monitoring-pipeline
docker rm growth-monitoring-pipeline
```

## Repository structure

The current repository is available both from inside the Docker and from the host machine. The results will be saved inside "results" directory, which is automatically created by the script. Below is the folder structure after a complete run session, with all information option variables enabled. 

- **model.pth** and **config.py** files inside "config" directory are in MMDetectionV3 format. 
- **annotations.json** file inside the "test_set" directory is in COCO annotation format.
- **annotations_tracked.json** file is automatically created by the script and is in COCO annotation format.
- **functions** directory contains all the auxiliary functions used in the pipeline.
- **config_paths.py** contains relevant paths for the execution of the python script in non-interactive mode. Please change accordingly to your needs.
- **config_variables.py** contains variables' initialization for the execution of the python script in non-interactive mode. Please change accordingly to your needs.
- **Clusters** directory contains images with the individual detected clusters cropped by bounding box.
- **Predictions** directory contains images with predictions from the instance segmentation model before and after filtering is applied.
- **Substrate** directory contains images with the substrate bounding box.
- **Tracked** directory contains images with tracking ID per cluster
- **Unracked** directory contains images with the random ID from the instance segmentation prediction.
- **clusters_relative_area.png** and **individual_relative_cluster_area.JPG** visualize the cluster area growth curves collectively and individually.
- **Cluster_Sizing.csv** contains the absolute size approximation and pixel size for each cluster detected in each image.
```
.
├── config/
│	├── instance_segmentation_model/
│	│	├── model.pth
│	│	└── config.py
│	├── reference_model/
│	│	├── model.pth
│	│	└── config.py
│	├── substrate_model/
│	│	├── model.pth
│	│	└── config.py
│	├── config_paths.py
│	├── config_variables.py
├── functions/
│	└── *.py
├── results/
│	├── Clusters/
│	│	└── *.JPG
│	├── Predictions/
│	│	├── before_filtering_*.JPG
│	│	└── after_filtering_*.JPG
│	├── Substrate/
│	│	└── *.JPG
│	├── Tracked/
│	│	└── *.JPG
│	├── Untracked/
│	│	└── *.JPG
│	├── absolute_cluster_area.JPG
│	├── individual_absolute_cluster_area.JPG
│	└── Cluster_Sizing.csv
├── test_set/
│	├── annotations/
│	│	├── annotations.json
│	│	└── annotations_tracked.json
│	└── images/
│		  └── *.JPG
├── Dockerfile
├── growth_monitoring_pipeline.ipynb
├── growth_monitoring_pipeline.py
├── README.md
└── requirements.txt
```
