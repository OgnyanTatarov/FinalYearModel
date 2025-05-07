import os

def setup_directories():
    """Create necessary directories for the ML module"""
    directories = [
        "data",
        "data/ml",
        "data/ml/models",
        "data/ml/encoders",
        "data/ml/plots"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    setup_directories() 