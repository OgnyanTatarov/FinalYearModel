import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import matplotlib.pyplot as plt

# Define paths
MODEL_PATH = "data/ml/models/revision_model.keras"
TOPIC_ENCODER_PATH = "data/ml/encoders/topic_encoder.pkl"
FORMAT_ENCODER_PATH = "data/ml/encoders/format_encoder.pkl"
PLOT_PATH = "data/ml/plots/training_history.png"

def build_revision_model(input_dim, topic_count, format_count):
    """Build the neural network model"""
    inputs = layers.Input(shape=(11,))  # Updated to expect 11 features
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    topic_output = layers.Dense(topic_count, activation='softmax', name='topic_output')(x)
    duration_output = layers.Dense(1, activation='relu', name='duration_output')(x)
    format_output = layers.Dense(format_count, activation='softmax', name='format_output')(x)

    model = Model(inputs, outputs=[topic_output, duration_output, format_output])
    model.compile(
        optimizer='adam',
        loss={
            'topic_output': 'sparse_categorical_crossentropy',
            'duration_output': 'mse',
            'format_output': 'sparse_categorical_crossentropy',
        },
        metrics={
            'topic_output': 'accuracy',
            'duration_output': 'mae',
            'format_output': 'accuracy',
        }
    )
    return model

def encode_dataset(data):
    """Encode the dataset for training"""
    import pandas as pd
    df = pd.DataFrame(data)

    # Create encoders
    topic_encoder = LabelEncoder()
    format_encoder = LabelEncoder()
    
    # Encode categorical variables
    df['topic_id'] = topic_encoder.fit_transform(df['topic'])
    df['format_id'] = format_encoder.fit_transform(df['format'])

    # Create one-hot encoding for subject (2 features)
    subject_encoder = LabelEncoder()
    df['subject_id'] = subject_encoder.fit_transform(df['subject'])
    subject_ohe = pd.get_dummies(df['subject_id'], prefix='subject')
    
    # Combine all features (2 + 5 + 4 = 11 features)
    features = pd.concat([
        subject_ohe,  # 2 features (one-hot encoded subject)
        df[['days_left', 'daily_time', 'topic_difficulty', 'confidence', 'performance']],  # 5 features
        df[['preferred_format_video', 'preferred_format_reading', 'preferred_format_quiz', 'preferred_format_practice']]  # 4 features
    ], axis=1)

    # Convert to numpy arrays
    X = features.values.astype(np.float32)
    y_topic = df['topic_id'].values.astype(np.int32)
    y_duration = df['duration'].values.astype(np.float32)
    y_format = df['format_id'].values.astype(np.int32)

    # Save encoders
    os.makedirs(os.path.dirname(TOPIC_ENCODER_PATH), exist_ok=True)
    joblib.dump(topic_encoder, TOPIC_ENCODER_PATH)
    joblib.dump(format_encoder, FORMAT_ENCODER_PATH)
    joblib.dump(subject_encoder, "data/ml/encoders/subject_encoder.pkl")

    print("Feature shapes:")
    print(f"Subject OHE: {subject_ohe.shape}")
    print(f"Numerical features: {df[['days_left', 'daily_time', 'topic_difficulty', 'confidence', 'performance']].shape}")
    print(f"Format features: {df[['preferred_format_video', 'preferred_format_reading', 'preferred_format_quiz', 'preferred_format_practice']].shape}")
    print(f"Final X shape: {X.shape}")

    return X, y_topic, y_duration, y_format, topic_encoder, format_encoder, subject_encoder

def train_model(X_train, y_topic, y_duration, y_format, input_dim, topic_count, format_count):
    """Train the model"""
    model = build_revision_model(input_dim, topic_count, format_count)

    # Train the model
    history = model.fit(
        X_train,
        {
            'topic_output': y_topic,
            'duration_output': y_duration,
            'format_output': y_format,
        },
        validation_split=0.2,
        epochs=30,
        batch_size=32
    )

    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    
    return history

def plot_training_history(history):
    """Plot the training history"""
    history_dict = history.history

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history_dict['topic_output_accuracy'], label='Topic Train Acc')
    plt.plot(history_dict['val_topic_output_accuracy'], label='Topic Val Acc')
    plt.plot(history_dict['format_output_accuracy'], label='Format Train Acc')
    plt.plot(history_dict['val_format_output_accuracy'], label='Format Val Acc')
    plt.title("Accuracy (Topic & Format)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history_dict['duration_output_mae'], label='Train MAE')
    plt.plot(history_dict['val_duration_output_mae'], label='Val MAE')
    plt.title("Duration MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history_dict['loss'], label='Train Loss')
    plt.plot(history_dict['val_loss'], label='Val Loss')
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH)
    plt.close()

def load_model():
    """Load the trained model and encoders"""
    model = tf.keras.models.load_model(MODEL_PATH)
    topic_encoder = joblib.load(TOPIC_ENCODER_PATH)
    format_encoder = joblib.load(FORMAT_ENCODER_PATH)
    subject_encoder = joblib.load("data/ml/encoders/subject_encoder.pkl")
    return model, topic_encoder, format_encoder, subject_encoder 