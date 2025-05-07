# ml_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder
import joblib

# === Build the model ===
def build_revision_model(input_dim, topic_count, format_count):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)

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

# === Encode dataset ===
def encode_dataset(data):
    import pandas as pd
    df = pd.DataFrame(data)

    topic_encoder = LabelEncoder()
    format_encoder = LabelEncoder()
    df['topic_id'] = topic_encoder.fit_transform(df['topic'])
    df['format_id'] = format_encoder.fit_transform(df['format'])

    subject_ohe = pd.get_dummies(df['subject'], prefix='subject')
    formats_ohe = df[['preferred_format_video', 'preferred_format_reading', 'preferred_format_quiz']]

    features = pd.concat([
        subject_ohe,
        df[['days_left', 'daily_time', 'topic_difficulty', 'confidence', 'performance']],
        formats_ohe
    ], axis=1)

    X = features.values.astype(np.float32)
    y_topic = df['topic_id'].values.astype(np.int32)
    y_duration = df['duration'].values.astype(np.float32)
    y_format = df['format_id'].values.astype(np.int32)

    joblib.dump(topic_encoder, "model/topic_encoder.pkl")
    joblib.dump(format_encoder, "model/format_encoder.pkl")

    return X, y_topic, y_duration, y_format, topic_encoder, format_encoder

# === Train the model ===
def train_model(X_train, y_topic, y_duration, y_format, input_dim, topic_count, format_count):
    model = build_revision_model(input_dim, topic_count, format_count)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath="model/exported_revision_model.keras",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1
    )

    history = model.fit(
        X_train,
        {
            'topic_output': y_topic,
            'duration_output': y_duration,
            'format_output': y_format,
        },
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[checkpoint_cb]
    )
    return history

# === Plot training history ===
def plot_training_history(history):
    import matplotlib.pyplot as plt
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
    plt.show()
