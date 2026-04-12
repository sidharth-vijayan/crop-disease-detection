# Crop Disease Detection Project

## Overview

This project implements an end-to-end crop disease detection system using deep learning models. It combines Convolutional Neural Networks (CNNs) for image classification, Long Short-Term Memory (LSTM) networks for temporal analysis, and a fusion model for enhanced predictions. The system includes a Python-based API, a web dashboard, and a Flutter mobile application for real-world deployment.

The project is structured in increments:
- **Increment 1**: CNN-based disease classification with domain adaptation
- **Increment 2**: LSTM for weather and yield prediction
- **Increment 3**: Fusion model combining CNN and LSTM outputs

## Features

- **Disease Classification**: Identifies 38 crop diseases from leaf images using EfficientNet-B4 backbone
- **Domain Adaptation**: Trained on PlantVillage dataset and augmented with PlantDoc field images for real-world robustness
- **Confidence Gating**: Production-ready inference with uncertainty thresholds
- **Grad-CAM Visualization**: Explainable AI for model predictions
- **Weather Integration**: LSTM model for weather-based yield predictions
- **Fusion Model**: Combines image and weather data for comprehensive insights
- **API and Dashboard**: FastAPI-based backend with Streamlit dashboard
- **Mobile App**: Flutter application for on-device disease detection
- **ONNX Export**: Models exported for cross-platform deployment

## Project Structure

```
crop-disease-detection/
├── api.py                          # FastAPI backend for disease prediction
├── dashboard.py                    # Streamlit dashboard for visualization
├── best_fusion.pth                 # Trained fusion model weights
├── best_lstm.pth                   # Trained LSTM model weights
├── best_model_v2.pth               # Trained CNN model weights (v2)
├── class_names.json                # 38 disease class names
├── fusion_config.json              # Fusion model configuration
├── lstm_config.json                # LSTM model configuration
├── dcws_scores.csv                 # Disease-Crop-Weather Scores
├── weather_features.csv            # Weather data for predictions
├── yield_model.json                # Yield prediction model
├── increment_1_cnn/                # CNN training and evaluation
│   ├── increment_1_cnn.ipynb       # Jupyter notebook for CNN training
│   ├── api.py                      # API script for CNN predictions
│   ├── best_model.pth              # Original CNN weights
│   ├── best_model_v2.pth           # Domain-adapted CNN weights
│   ├── crop_disease_cnn.onnx       # ONNX model (original)
│   ├── crop_disease_cnn_v2.onnx    # ONNX model (v2)
│   ├── class_names.json            # Class names
│   ├── PlantDoc-Dataset/           # PlantDoc dataset
│   └── plantvillage dataset/       # PlantVillage dataset
├── crop_disease_ews/               # Flutter mobile application
│   ├── lib/                        # Dart source code
│   ├── android/                    # Android platform code
│   ├── ios/                        # iOS platform code
│   └── pubspec.yaml                # Flutter dependencies
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.8+
- Flutter SDK (for mobile app)
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd crop-disease-detection
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install fastapi uvicorn streamlit onnx onnxruntime
   pip install albumentations scikit-learn matplotlib seaborn tqdm pillow opencv-python
   ```

4. **Setup Flutter app** (optional, for mobile development):
   ```bash
   cd crop_disease_ews
   flutter pub get
   flutter run  # For testing on device/emulator
   ```

## Usage

### Running the API

```bash
python api.py
```

The API will start on `http://localhost:8000`. Use endpoints like `/predict` for disease classification.

### Running the Dashboard

```bash
streamlit run dashboard.py
```

Access the dashboard at `http://localhost:8501` for visualizations and predictions.

### Training Models

Open the Jupyter notebooks in `increment_1_cnn/` and run cells sequentially:

1. `increment_1_cnn.ipynb`: Train CNN model with domain adaptation
2. `increment_2_lstm.ipynb`: Train LSTM for weather/yield prediction
3. `increment_3_fusion.ipynb`: Train fusion model

### Mobile App

Navigate to `crop_disease_ews/` and run:

```bash
flutter run
```

## Model Details

### CNN Model (Increment 1)
- **Architecture**: EfficientNet-B4 with custom classifier head
- **Input**: 380x380 RGB images
- **Output**: 38 disease classes
- **Training**: PlantVillage + PlantDoc datasets with augmentation
- **Accuracy**: ~96-98% on PlantVillage test set

### LSTM Model (Increment 2)
- **Purpose**: Weather-based yield prediction
- **Input**: Time-series weather data
- **Output**: Yield forecasts

### Fusion Model (Increment 3)
- **Purpose**: Combine CNN and LSTM outputs
- **Input**: Image embeddings + weather features
- **Output**: Enhanced disease and yield predictions

## Datasets

- **PlantVillage**: 54,305 images across 38 classes (controlled lab conditions)
- **PlantDoc**: Field images for domain adaptation (real-world robustness)

## Deployment

Models are exported to ONNX format for cross-platform deployment. Use the `.onnx` files in production environments.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.