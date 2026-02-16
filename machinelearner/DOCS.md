# Home Assistant Machine Learner Addon

This addon provides a FastAPI-based service for training and making predictions with LightGBM machine learning models, designed for use with Home Assistant.

## Features

- Train LightGBM models with provided data
- Make predictions using trained models 
- In-memory model storage
- RESTful API endpoints
- Home Assistant addon compatible

## API Endpoints

### Train Model
```
POST /train
```

**Request Body:**
```json
{
  "model_name": "string",
  "dataframe": "csv string",
  "target_column": "string"
}
```

**Response:**
```json
{
  "model_name": "string",
  "status": "string"
}
```

### Predict
```
POST /predict
```

**Request Body:**
```json
{
  "model_name": "string",
  "dataframe": "csv string"
}
```

**Response:**
```json
{
  "predictions": [number]
}
```

### Health Check
```
GET /health
```

## Usage

1. Start the addon in Home Assistant
2. Send training data to `/train` endpoint
3. Use trained model for predictions via `/predict` endpoint

## Example Usage

### Training
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "temperature_predictor",
    "dataframe": "temp,humidity,pressure,target\n20,60,1013,22\n21,65,1012,23\n22,70,1011,24",
    "target_column": "target"
  }'
```

### Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "temperature_predictor",
    "dataframe": "temp,humidity,pressure\n25,75,1010"
  }'
```

## Requirements

- Python 3.9+
- FastAPI
- pandas
- LightGBM
