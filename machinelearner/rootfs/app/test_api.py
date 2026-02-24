#!/usr/bin/env python3
"""
Test script to verify the API functionality
"""
import requests
import pandas as pd
import numpy as np
from io import StringIO

# Create sample data for testing
def create_sample_data():
    # Create a simple dataset for training
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'target': [3, 6, 9, 12, 15]
    }
    df = pd.DataFrame(data)
    return df

# Test the API
def test_api():
    base_url = "http://localhost:14760"
    
    # Create sample data
    df = create_sample_data()
    csv_data = df.to_csv(index=False)
    
    print("Testing API endpoints...")
    
    # Test training
    train_data = {
        "model_name": "test_model",
        "dataframe": csv_data,
        "target_column": "target"
    }
    
    try:
        response = requests.post(f"{base_url}/train", json=train_data)
        print(f"Training response: {response.status_code}")
        print(f"Training result: {response.json()}")
    except Exception as e:
        print(f"Error during training: {e}")
    
    # Test is_trained endpoint - should return True for the trained model
    try:
        response = requests.get(f"{base_url}/is_trained?model_name=test_model")
        print(f"Is trained response: {response.status_code}")
        print(f"Is trained result: {response.json()}")
    except Exception as e:
        print(f"Error during is_trained check: {e}")
    
    # Test is_trained endpoint - should return False for a non-existent model
    try:
        response = requests.get(f"{base_url}/is_trained?model_name=non_existent_model")
        print(f"Is trained (non-existent) response: {response.status_code}")
        print(f"Is trained (non-existent) result: {response.json()}")
    except Exception as e:
        print(f"Error during is_trained check (non-existent): {e}")
    
    # For prediction, we need to use just the feature columns (excluding target)
    # Create a new dataframe with only the features to predict on
    prediction_df = df[['feature1', 'feature2']]  # Only features, no target
    prediction_csv = prediction_df.to_csv(index=False)
    
    # Test prediction
    predict_data = {
        "model_name": "test_model",
        "dataframe": prediction_csv
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=predict_data)
        print(f"Prediction response: {response.status_code}")
        print(f"Prediction result: {response.json()}")
    except Exception as e:
        print(f"Error during prediction: {e}")
    
    # Test health
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code}")
        print(f"Health result: {response.json()}")
    except Exception as e:
        print(f"Error during health check: {e}")

if __name__ == "__main__":
    test_api()