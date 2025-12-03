"""
FastAPI application for sentiment analysis
Provides REST API endpoints for sentiment prediction with MongoDB logging and Redis caching
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import torch
import logging
from datetime import datetime
import os
import json
import hashlib

# Database connections
import redis
from pymongo import MongoClient

from src.inference import load_model, predict_sentiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="BERT-based sentiment analysis for app reviews with MLOps integration",
    version="2.1.0"
)

# Global variables
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/final_model")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:sentimentpass123@mongodb:27017/")
REDIS_HOST = os.getenv("REDIS_HOST", "redis-cache")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

model = None
tokenizer = None
device = None
mongo_client = None
db = None
redis_client = None


class PredictionRequest(BaseModel):
    """Request model for sentiment prediction"""
    text: str


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    texts: List[str]


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    text: str
    sentiment: str
    confidence: float
    probabilities: dict
    timestamp: str
    cached: bool = False


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    mongo_connected: bool
    redis_connected: bool
    version: str
    timestamp: str


def get_db():
    """Get MongoDB database connection"""
    global mongo_client, db
    if mongo_client is None:
        try:
            mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            db = mongo_client.sentiment_logs
            # Check connection
            mongo_client.server_info()
            logger.info("Connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            mongo_client = None
            db = None
    return db


def get_redis():
    """Get Redis connection"""
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
            # Check connection
            redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            redis_client = None
    return redis_client


def log_prediction(text: str, result: dict, endpoint: str):
    """Log prediction to MongoDB in background"""
    database = get_db()
    if database is not None:
        try:
            log_entry = {
                "text": text,
                "sentiment": result['sentiment'],
                "confidence": result['confidence'],
                "probabilities": result['probabilities'],
                "endpoint": endpoint,
                "timestamp": datetime.now(),
                "model_version": "bert-base-uncased"
            }
            database.predictions.insert_one(log_entry)
        except Exception as e:
            logger.error(f"Failed to log to MongoDB: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    global model, tokenizer, device
    
    logger.info("Starting up Sentiment Analysis API...")
    
    # Load Model
    try:
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading model from {MODEL_PATH}...")
            model, tokenizer, device = load_model(MODEL_PATH)
            logger.info(f"Model loaded successfully on device: {device}")
        else:
            logger.warning(f"Model not found at {MODEL_PATH}. API will start but predictions will fail.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

    # Initialize connections
    get_db()
    get_redis()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    db_status = False
    try:
        if get_db() is not None:
            mongo_client.server_info()
            db_status = True
    except:
        pass

    redis_status = False
    try:
        r = get_redis()
        if r and r.ping():
            redis_status = True
    except:
        pass

    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        mongo_connected=db_status,
        redis_connected=redis_status,
        version="2.1.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Predict sentiment for a single text"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check Cache
    cache = get_redis()
    cache_key = f"sentiment:{hashlib.md5(request.text.encode()).hexdigest()}"
    
    if cache:
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info("Cache hit")
            result = json.loads(cached_result)
            return PredictionResponse(
                text=request.text,
                sentiment=result['sentiment'],
                confidence=result['confidence'],
                probabilities=result['probabilities'],
                timestamp=datetime.now().isoformat(),
                cached=True
            )

    try:
        # Predict
        result = predict_sentiment(request.text, model, tokenizer, device)
        
        # Cache result (expire in 1 hour)
        if cache:
            cache.setex(cache_key, 3600, json.dumps(result))
        
        # Log to MongoDB in background
        background_tasks.add_task(log_prediction, request.text, result, "/predict")
        
        return PredictionResponse(
            text=result['text'],
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            timestamp=datetime.now().isoformat(),
            cached=False
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=List[PredictionResponse])
async def batch_predict(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """Predict sentiment for multiple texts"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Batch size limit exceeded (100)")
    
    results = []
    cache = get_redis()
    
    for text in request.texts:
        try:
            # Check cache for each item
            cache_key = f"sentiment:{hashlib.md5(text.encode()).hexdigest()}"
            cached = False
            
            if cache:
                cached_res = cache.get(cache_key)
                if cached_res:
                    res = json.loads(cached_res)
                    cached = True
                else:
                    res = predict_sentiment(text, model, tokenizer, device)
                    cache.setex(cache_key, 3600, json.dumps(res))
            else:
                res = predict_sentiment(text, model, tokenizer, device)
            
            results.append(PredictionResponse(
                text=res['text'],
                sentiment=res['sentiment'],
                confidence=res['confidence'],
                probabilities=res['probabilities'],
                timestamp=datetime.now().isoformat(),
                cached=cached
            ))
            
            # Log sample (not all to avoid flooding)
            if not cached:
                background_tasks.add_task(log_prediction, text, res, "/batch")
                
        except Exception as e:
            logger.error(f"Error processing text '{text[:20]}...': {e}")
            
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
