import json
import numpy as np
from sklearn.model_selection import train_test_split
from .model import encode_dataset, train_model, plot_training_history, load_model, build_revision_model
from .dataset import load_dataset
from .setup import setup_directories
import random
import os
import pickle

# Define paths
TRAINING_DATA_PATH = "data/ml/training_data.json"
METRICS_PATH = "data/ml/training_metrics.json"

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
                
                # Create training sample with all features
                sample = {
                    "subject": subject_id,  # Use subject_id (e.g., "math", "physics")
                    "days_left": days_left,
                    "daily_time": daily_time,
                    "topic": topic_id,  # Use topic_id (e.g., "algebra", "calculus")
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
    os.makedirs(os.path.dirname(TRAINING_DATA_PATH), exist_ok=True)
    with open(TRAINING_DATA_PATH, "w") as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Generated {len(training_data)} training samples")
    return training_data

def prepare_training_data():
    """Prepare training data from our dataset"""
    # Generate synthetic training data
    print("Generating training data...")
    data = generate_training_data(num_samples=2000)  # Generate 2000 samples
    
    # === Encode data ===
    print("Encoding data...")
    X, y_topic, y_duration, y_format, topic_enc, format_enc, subject_enc = encode_dataset(data)
    
    # === Split data ===
    print("Splitting data...")
    X_train, X_val, yt_train, yt_val, yd_train, yd_val, yf_train, yf_val = train_test_split(
        X, y_topic, y_duration, y_format, test_size=0.2, random_state=42
    )
    
    return X_train, X_val, yt_train, yt_val, yd_train, yd_val, yf_train, yf_val, X.shape[1], len(topic_enc.classes_), len(format_enc.classes_)

def evaluate_model(model, X_val, yt_val, yd_val, yf_val, topic_encoder, format_encoder):
    """Evaluate the model on validation data"""
    # Make predictions
    topic_pred, duration_pred, format_pred = model.predict(X_val)
    
    # Get the most likely topics and formats
    topic_idx = np.argmax(topic_pred, axis=1)
    format_idx = np.argmax(format_pred, axis=1)
    
    # Decode predictions
    topics = topic_encoder.inverse_transform(topic_idx)
    formats = format_encoder.inverse_transform(format_idx)
    
    # Calculate metrics
    topic_accuracy = float(np.mean(topics == topic_encoder.inverse_transform(yt_val)))
    format_accuracy = float(np.mean(formats == format_encoder.inverse_transform(yf_val)))
    duration_mae = float(np.mean(np.abs(duration_pred.flatten() - yd_val)))
    
    print("\nValidation Metrics:")
    print(f"Topic Accuracy: {topic_accuracy:.2%}")
    print(f"Format Accuracy: {format_accuracy:.2%}")
    print(f"Duration MAE: {duration_mae:.2f} hours")
    
    return {
        "topic_accuracy": topic_accuracy,
        "format_accuracy": format_accuracy,
        "duration_mae": duration_mae
    }

def train():
    # Create necessary directories
    os.makedirs('data/ml/models', exist_ok=True)
    os.makedirs('data/ml/encoders', exist_ok=True)
    os.makedirs('data/ml/plots', exist_ok=True)

    # Generate training data
    data = generate_training_data()
    print(f"Generated {len(data)} training samples")

    # Encode the data
    X, y_topic, y_duration, y_format, topic_encoder, format_encoder, subject_encoder = encode_dataset(data)
    print("Encoding data...")
    print("Feature shapes:")
    print(f"Subject OHE: {X[:, :2].shape}")
    print(f"Numerical features: {X[:, 2:7].shape}")
    print(f"Format features: {X[:, 7:].shape}")
    print(f"Final X shape: {X.shape}")

    # Get the number of unique topics and formats
    topic_count = len(topic_encoder.classes_)
    format_count = len(format_encoder.classes_)

    # Split the data
    print("Splitting data...")
    X_train, X_val, y_topic_train, y_topic_val, y_duration_train, y_duration_val, y_format_train, y_format_val = train_test_split(
        X, y_topic, y_duration, y_format, test_size=0.2, random_state=42
    )

    # Build and train the model
    print("\nTraining model...")
    model = build_revision_model(X.shape[1], topic_count, format_count)
    history = model.fit(
        X_train,
        {
            'topic_output': y_topic_train,
            'duration_output': y_duration_train,
            'format_output': y_format_train
        },
        validation_data=(
            X_val,
            {
                'topic_output': y_topic_val,
                'duration_output': y_duration_val,
                'format_output': y_format_val
            }
        ),
        epochs=30,
        batch_size=50
    )

    # Save the best model
    model.save('data/ml/models/revision_model.keras')
    with open('data/ml/encoders/topic_encoder.pkl', 'wb') as f:
        pickle.dump(topic_encoder, f)
    with open('data/ml/encoders/format_encoder.pkl', 'wb') as f:
        pickle.dump(format_encoder, f)
    with open('data/ml/encoders/subject_encoder.pkl', 'wb') as f:
        pickle.dump(subject_encoder, f)

    # Save training metrics
    metrics = {
        'topic_accuracy': float(history.history['val_topic_output_accuracy'][-1]),
        'format_accuracy': float(history.history['val_format_output_accuracy'][-1]),
        'duration_mae': float(history.history['val_duration_output_mae'][-1])
    }
    with open('data/ml/training_metrics.json', 'w') as f:
        json.dump(metrics, f)

    print("\nTraining completed!")
    print(f"Topic accuracy: {metrics['topic_accuracy']:.2%}")
    print(f"Format accuracy: {metrics['format_accuracy']:.2%}")
    print(f"Duration MAE: {metrics['duration_mae']:.2f} hours")

if __name__ == "__main__":
    train() 