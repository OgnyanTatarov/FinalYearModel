import json
import random
import numpy as np
from dataset import load_dataset

def generate_training_data(num_samples=1000):
    """Generate synthetic training data based on our dataset"""
    dataset = load_dataset()
    training_data = []
    
    # Generate data for each subject and topic
    for subject_id, subject in dataset["subjects"].items():
        for topic_id, topic in subject["topics"].items():
            # Generate multiple samples for each topic
            for _ in range(num_samples // len(dataset["subjects"]) // len(subject["topics"])):
                # Generate random user preferences
                daily_time = round(random.uniform(0.5, 4.0), 1)  # 30 mins to 4 hours
                days_left = random.randint(1, 30)
                confidence = random.randint(1, 5)
                performance = round(random.uniform(40, 95), 1)
                
                # Generate format preferences
                available_formats = topic["formats"]
                preferred_formats = random.sample(available_formats, min(2, len(available_formats)))
                
                # Calculate difficulty based on topic and user performance
                base_difficulty = topic["difficulty"]
                adjusted_difficulty = max(1, min(5, base_difficulty + random.randint(-1, 1)))
                
                # Calculate duration based on topic's estimated time and user factors
                base_duration = topic["estimated_time"]
                # Adjust duration based on confidence and performance
                confidence_factor = (6 - confidence) / 5  # Lower confidence = longer duration
                performance_factor = (100 - performance) / 100  # Lower performance = longer duration
                duration = round(base_duration * (1 + confidence_factor * 0.5 + performance_factor * 0.5), 1)
                
                # Select a format based on preferences and availability
                format_used = random.choice(preferred_formats)
                
                # Create training sample
                sample = {
                    "subject": subject_id,
                    "days_left": days_left,
                    "daily_time": daily_time,
                    "topic": topic_id,
                    "topic_difficulty": adjusted_difficulty,
                    "confidence": confidence,
                    "performance": performance,
                    "preferred_format_video": int("video" in preferred_formats),
                    "preferred_format_reading": int("reading" in preferred_formats),
                    "preferred_format_quiz": int("quiz" in preferred_formats),
                    "preferred_format_practice": int("practice" in preferred_formats),
                    "duration": duration,
                    "format": format_used
                }
                training_data.append(sample)
    
    # Shuffle the data
    random.shuffle(training_data)
    
    # Save to file
    with open("data/synthetic_training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Generated {len(training_data)} training samples")
    return training_data

if __name__ == "__main__":
    generate_training_data() 