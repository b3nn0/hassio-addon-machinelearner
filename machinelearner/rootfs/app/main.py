import pickle
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import numpy as np
from io import StringIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Home Assistant Machine Learner",
              description="API for training and predicting with LightGBM models")

# In-memory storage for models
models: Dict[str, Any] = {}

class TrainRequest(BaseModel):
    model_name: str
    dataframe: str  # Serialized dataframe as CSV string
    target_column: str

class PredictRequest(BaseModel):
    model_name: str
    dataframe: str  # Serialized dataframe as CSV string

class TrainResponse(BaseModel):
    model_name: str
    status: str

class PredictResponse(BaseModel):
    predictions: list

class IsTrainedResponse(BaseModel):
    model_name: str
    is_trained: bool

@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """Train a LightGBM model with the provided data"""
    try:
        logger.info(f"Training model '{request.model_name}'")
        
        # Parse the CSV string into a pandas DataFrame
        df = pd.read_csv(StringIO(request.dataframe))
        
        # Validate that target column exists
        if request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found in DataFrame")
        
        # Sort by column name
        df = df.reindex(sorted(df.columns), axis=1)
        
        # Separate features and target
        x = df.drop(columns=[request.target_column])
        y = df[request.target_column]
        
       
        # Create LightGBM dataset
        train_data = lgb.Dataset(x, label=y)
        
        # Define parameters for LightGBM
        params = {
            #'objective': 'regression',
            #'metric': 'rmse',
            #'boosting_type': 'gbdt',
            #'num_leaves': 31,
            #'learning_rate': 0.05,
            #'feature_fraction': 0.9
        }
        
        # Train the model
        model = lgb.train(params, train_data)
        
        # Store the model in memory
        models[request.model_name] = model
        
        logger.info(f"Model '{request.model_name}' trained successfully")
        
        return TrainResponse(
            model_name=request.model_name,
            status="Model trained successfully"
        )
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Make predictions using a trained model"""
    try:
        logger.info(f"Making prediction with model '{request.model_name}'")
        
        # Check if model exists
        if request.model_name not in models:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")
        
        # Parse the CSV string into a pandas DataFrame
        df = pd.read_csv(StringIO(request.dataframe))

        df = df.reindex(sorted(df.columns), axis=1)
        
        # Get the trained model
        model = models[request.model_name]
        
        # Make predictions - just let it fail naturally with a proper error message
        # This will catch most feature count mismatches
        try:
            predictions = model.predict(df)
        except Exception as e:
            error_str = str(e)
            if "number of features" in error_str.lower() or "shape" in error_str.lower():
                raise HTTPException(
                    status_code=400,
                    detail="Data format mismatch: The number of features in your data doesn't match the trained model. Please ensure your prediction data has the same features as the training data."
                )
            else:
                raise e
        
        # Convert to list for JSON serialization
        predictions_list = predictions.tolist()
        
        logger.info(f"Prediction completed for model '{request.model_name}'")
        
        return PredictResponse(predictions=predictions_list)
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_count": len(models)}

@app.get("/is_trained", response_model=IsTrainedResponse)
async def is_trained(model_name: str):
    """Check if a model with the specified name is trained"""
    is_trained = model_name in models
    return IsTrainedResponse(
        model_name=model_name,
        is_trained=is_trained
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=14760)