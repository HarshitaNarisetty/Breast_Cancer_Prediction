# Breast Cancer Prediction using Logistic Regression and Streamlit

This project is a breast cancer prediction model that uses Logistic Regression to classify tumors as malignant or benign based on input features. The application is built using Streamlit for an interactive user experience.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)



## Overview

This project uses machine learning techniques to classify breast cancer cases. The Logistic Regression model is trained on a dataset of cell nuclei features extracted from breast cancer biopsy images. The Streamlit application provides a user-friendly interface to predict results.

## Features

- Logistic Regression model for classification
- Interactive UI using Streamlit
- Real-time prediction using user input
- Model accuracy and classification report displayed

## Dataset

- Dataset: **Breast Cancer Wisconsin (Diagnostic) Dataset**

- Source: [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

- The dataset contains features computed from a digitized image of a breast mass.

- Usage

1. Train the model:
   ```bash
   python model.py
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Enter the input values and get predictions.

## Model Training

- The model is built using **Logistic Regression**.
- Data is standardized using **StandardScaler**.
- The model is evaluated using **Accuracy Score** and **Classification Report**.

## Results

- Achieved accuracy of approximately 0.97/1.00 on test data.\*\*
- Detailed classification report is displayed during training.

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

