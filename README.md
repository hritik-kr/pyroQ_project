# 🔥 PyroQ: Quantum-Enhanced Wildfire Detection

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](requirements.txt)

**Detect wildfires using quantum-classical hybrid models on satellite imagery**

![Demo](assets/pyroq_demo.gif) 

## 🚀 Features
- **Quantum CNN** for thermal anomaly detection
- **FastAPI** backend with JWT authentication
- **GeoTIFF processing** for satellite data
- **Docker** deployment with GPU support

## 📦 Installation

git clone https://github.com/username/pyroQ_project.git
cd pyroQ_project
pip install -r requirements.txt


## 🖥️ Usage

from src.hybrid import PyroQModel
model = PyroQModel.load("models/pyroq.pt")
predictions = model.detect("thermal_image.tif")


📊 Results

Model	Accuracy	Precision	Recall
Quantum Hybrid	92.3%	89.5%	94.1%
Classical CNN	87.6%	85.2%	88.9%
