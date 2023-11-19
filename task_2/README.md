# Wine Quality Prediction Project

## Overview
This project focuses on analyzing, modeling, and predicting the quality of wine based on various properties. 
It includes scripts for exploratory data analysis, feature engineering, model training, and batch inference.

## Hugging face spaces
Wine predictor: https://huggingface.co/spaces/martenb/wine
Daily predictions monitor: https://huggingface.co/spaces/martenb/wine-monitor

## Project Structure
- `wine-eda-and-backfill-feature-group.ipynb`: Jupyter notebook for exploratory data analysis and feature engineering.
- `wine-training-pipeline.ipynb`: Jupyter notebook for training machine learning models.
- `daily-wine-feature-pipeline.py`: Python script for daily feature engineering pipeline.
- `wine-batch-inference-pipeline.py`: Python script for batch inference pipeline.
- `../modal_daily-wine-feature-pipeline.sh`: Bash script to more easily run the daily feature engineering process.
- `../modal_wine-batch-inference-pipeline.sh`: Bash script to more easily run the batch inference process.

## How to Run the Project
### Prerequisites
Ensure you have Python and the necessary libraries installed. It's recommended to use a virtual environment.

### Running Jupyter Notebooks
1. Open Jupyter Notebook or Jupyter Lab.
2. Run the notebooks in the following order:
   - `wine-eda-and-backfill-feature-group.ipynb`
   - `wine-training-pipeline.ipynb`

### Running Python Scripts
1. Run the Python scripts via the bash scripts directly using:
   ```bash
   chmod +x ../modal_daily-wine-feature-pipeline.sh
   chmod +x ../modal_wine-batch-inference-pipeline.sh
   ../modal_daily-wine-feature-pipeline.sh
   ../modal_wine-batch-inference-pipelin.sh
   ```
