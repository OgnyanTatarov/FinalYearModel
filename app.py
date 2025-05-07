from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import tensorflow as tf
import joblib

# Load model and encoders
model = tf.keras.models.load_model("model/exported_revision_model.keras")
topic_encoder = joblib.load("model/topic_encoder.pkl")
format_encoder = joblib.load("model/format_encoder.pkl")

subject_list = ["Math", "History", "Physics"]
format_list = ["video", "reading", "quiz"]

class TopicInfo(BaseModel):
    topic: str
    difficulty: int
    confidence: str
    performance: float

class MultiDayInput(BaseModel):
    subject: str
    days_left: int
    daily_time: float
    topics: List[TopicInfo]
    preferred_formats: List[str]

app = FastAPI()

@app.post("/generate_plan")
def generate_plan(input_data: MultiDayInput):
    plan = []
    subject_ohe = [1 if input_data.subject == s else 0 for s in subject_list]
    format_ohe = [1 if f in input_data.preferred_formats else 0 for f in format_list]

    def score_topic(t):
        conf = {"low": 1, "medium": 3, "high": 5}.get(t.confidence, 3)
        perf = t.performance
        score = (6 - conf) + (70 - perf) / 10
        return max(score, 0.1)

    topic_scores = [(t, score_topic(t)) for t in input_data.topics]
    total_score = sum(score for _, score in topic_scores)

    day_cursor = 1
    for topic_info, score in topic_scores:
        time_allocation = (score / total_score) * input_data.days_left * input_data.daily_time
        chunks = max(1, round(time_allocation / input_data.daily_time))

        for _ in range(chunks):
            conf_val = {"low": 1, "medium": 3, "high": 5}[topic_info.confidence]
            features = np.array([
                *subject_ohe,
                input_data.days_left,
                input_data.daily_time,
                topic_info.difficulty,
                conf_val,
                topic_info.performance,
                *format_ohe
            ]).reshape(1, -1)

            print("All subjects (from encoder):", subject_list)
            print("Subject features:", subject_ohe)
            print("Final features shape:", features.shape)
            print("Final features:", features)

            preds = model.predict(features)
            pred_topic = topic_encoder.inverse_transform([np.argmax(preds[0])])[0]
            pred_format = format_encoder.inverse_transform([np.argmax(preds[2])])[0]
            pred_duration = float(preds[1][0][0])

            plan.append({
                "day": day_cursor,
                "topic": pred_topic,
                "format": pred_format,
                "duration": round(min(pred_duration, input_data.daily_time), 2)
            })

            day_cursor += 1
            if day_cursor > input_data.days_left:
                day_cursor = 1

    return plan
