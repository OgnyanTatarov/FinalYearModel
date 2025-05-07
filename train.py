# train.py
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from ml_model import encode_dataset, train_model, plot_training_history
from generate_training_data import generate_training_data
from dataset import load_dataset

def prepare_training_data():
    """Prepare training data from our dataset"""
    # Generate synthetic training data
    print("Generating training data...")
    data = generate_training_data(num_samples=2000)  # Generate 2000 samples
    
    # === Encode data ===
    print("Encoding data...")
    X, y_topic, y_duration, y_format, topic_enc, format_enc = encode_dataset(data)
    
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
    topic_accuracy = np.mean(topics == topic_encoder.inverse_transform(yt_val))
    format_accuracy = np.mean(formats == format_encoder.inverse_transform(yf_val))
    duration_mae = np.mean(np.abs(duration_pred.flatten() - yd_val))
    
    print("\nValidation Metrics:")
    print(f"Topic Accuracy: {topic_accuracy:.2%}")
    print(f"Format Accuracy: {format_accuracy:.2%}")
    print(f"Duration MAE: {duration_mae:.2f} hours")
    
    return {
        "topic_accuracy": topic_accuracy,
        "format_accuracy": format_accuracy,
        "duration_mae": duration_mae
    }

def main():
    # === Prepare data ===
    X_train, X_val, yt_train, yt_val, yd_train, yd_val, yf_train, yf_val, input_dim, topic_classes, format_classes = prepare_training_data()
    
    # === Train the model ===
    print("\nTraining model...")
    history = train_model(X_train, yt_train, yd_train, yf_train, input_dim, topic_classes, format_classes)
    
    # === Plot training history ===
    plot_training_history(history)
    
    # === Evaluate model ===
    model = tf.keras.models.load_model("model/exported_revision_model.keras")
    topic_encoder = joblib.load("model/topic_encoder.pkl")
    format_encoder = joblib.load("model/format_encoder.pkl")
    
    metrics = evaluate_model(model, X_val, yt_val, yd_val, yf_val, topic_encoder, format_encoder)
    
    # Save metrics
    with open("model/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nModel training complete. Best version saved in: model/exported_revision_model.keras")
    print("Training metrics saved in: model/training_metrics.json")

if __name__ == "__main__":
    main()
