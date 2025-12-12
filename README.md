# Satellite-Image-Segmentation
Satellite Image Segmentation using U-Net is a deep learning project focused on extracting meaningful land-cover information from satellite imagery. The goal is to automatically segment regions such as roads, vegetation, buildings, and other terrain types using a U-Net architecture.

This project includes the complete workflow from dataset preparation to model training, evaluation, and visualization.

🔍 Features

U-Net architecture for pixel-wise segmentation

Preprocessing pipeline (normalization, resizing, mask alignment)

Data augmentation for improving generalization

Training on satellite imagery datasets (e.g., DeepGlobe / ISPRS / custom)

Evaluation using IoU, Dice Score, and F1

Visualization of predictions vs ground truth

Clean, modular, and reusable code

🧠 Model Overview

U-Net is used because it is highly effective for segmentation tasks with limited data.
Its encoder–decoder structure helps capture both the context and fine-grained details in satellite images.

📂 Project Structure

data/ — images and segmentation masks

src/ — training scripts, model architecture, utilities

notebooks/ — experiments and EDA

outputs/ — trained models and prediction samples

🚀 Use Cases

Land-cover mapping

Urban planning

Disaster analysis (floods, fires, damaged areas)

Agriculture and vegetation tracking

Environmental monitoring

📊 Results

The model successfully segments satellite images with strong performance across classes, producing accurate pixel-level masks.

🛠️ Tech Stack

Python, NumPy, OpenCV

TensorFlow / PyTorch

Matplotlib

U-Net (custom implementation)
