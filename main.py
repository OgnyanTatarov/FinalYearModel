from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from typing import List, Dict, Any
from pydantic import BaseModel
from .ml.dataset import get_topic_info, get_available_topics, get_available_subjects
from .ml.model import load_model

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and encoders
try:
    model, topic_encoder, format_encoder, subject_encoder = load_model()
    print("ML model loaded successfully")
except Exception as e:
    print(f"Error loading ML model: {e}")
    model = None
    topic_encoder = None
    format_encoder = None
    subject_encoder = None

class RevisionRequest(BaseModel):
    subject: str
    days_left: int
    daily_time: float
    topic_difficulty: int
    confidence: int
    performance: float
    preferred_format_video: bool
    preferred_format_reading: bool
    preferred_format_quiz: bool
    preferred_format_practice: bool

class ContentItem(BaseModel):
    title: str
    duration: float
    url: str
    questions: int = None
    problems: int = None

class TopicContent(BaseModel):
    video: List[ContentItem] = []
    reading: List[ContentItem] = []
    quiz: List[ContentItem] = []
    practice: List[ContentItem] = []

class RevisionResponse(BaseModel):
    topic: str
    topic_name: str
    duration: float
    format: str
    difficulty: int
    prerequisites: List[str]
    content: TopicContent
    estimated_time: float

def prepare_features(request: RevisionRequest) -> np.ndarray:
    """Prepare input features for the model, matching the model's expected input shape."""
    # Get subject ID and create one-hot encoding
    subject_id = subject_encoder.transform([request.subject])[0]
    subject_features = np.zeros(2, dtype=np.float32)  # 2 possible subjects
    subject_features[subject_id] = 1.0
    
    # Combine all features in the same order as training (11 features total)
    features = np.concatenate([
        subject_features,  # 2 features (one-hot encoded subject)
        np.array([
            request.days_left,
            request.daily_time,
            request.topic_difficulty,
            request.confidence,
            request.performance,
            int(request.preferred_format_video),
            int(request.preferred_format_reading),
            int(request.preferred_format_quiz),
            int(request.preferred_format_practice)
        ], dtype=np.float32)  # 9 features
    ])
    
    print("Subject ID:", subject_id)
    print("Subject features:", subject_features)
    print("Final features shape:", features.shape)
    print("Final features:", features)
    
    return features.reshape(1, -1)

@app.post("/api/revision/suggest", response_model=RevisionResponse)
async def suggest_revision(request: RevisionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
        
    try:
        # Prepare input features
        features = prepare_features(request)
        
        # Make prediction
        topic_pred, duration_pred, format_pred = model.predict(features)
        
        # Get the most likely topic and format
        topic_idx = np.argmax(topic_pred[0])
        format_idx = np.argmax(format_pred[0])
        
        # Decode predictions
        topic = topic_encoder.inverse_transform([topic_idx])[0]
        format = format_encoder.inverse_transform([format_idx])[0]
        
        # Get detailed topic information
        topic_info = get_topic_info(request.subject, topic)
        if not topic_info:
            raise HTTPException(status_code=404, detail="Topic not found")
        
        # Prepare content items
        content = TopicContent()
        for content_type, items in topic_info["content"].items():
            for item in items:
                content_item = ContentItem(
                    title=item["title"],
                    duration=item["duration"],
                    url=item["url"],
                    questions=item.get("questions"),
                    problems=item.get("problems")
                )
                getattr(content, content_type).append(content_item)
        
        return RevisionResponse(
            topic=topic,
            topic_name=topic_info["name"],
            duration=float(duration_pred[0][0]),
            format=format,
            difficulty=topic_info["difficulty"],
            prerequisites=topic_info["prerequisites"],
            content=content,
            estimated_time=topic_info["estimated_time"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/revision/topics")
async def get_topics(subject: str):
    try:
        topics = get_available_topics(subject)
        if not topics:
            raise HTTPException(status_code=404, detail="No topics found for this subject")
        return {"topics": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/revision/subjects")
async def get_subjects():
    try:
        subjects = get_available_subjects()
        return {"subjects": subjects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/revision/formats")
async def get_formats(subject: str, topic: str):
    try:
        topic_info = get_topic_info(subject, topic)
        if not topic_info:
            raise HTTPException(status_code=404, detail="Topic not found")
        return {"formats": topic_info["formats"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 