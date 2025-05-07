import json
import os

# Comprehensive study dataset
STUDY_DATASET = {
    "subjects": {
        "math": {
            "name": "Mathematics",
            "topics": {
                "algebra": {
                    "name": "Algebra",
                    "difficulty": 3,
                    "estimated_time": 2.0,
                    "prerequisites": ["arithmetic"],
                    "formats": ["video", "quiz", "practice"],
                    "content": {
                        "video": [
                            {
                                "title": "Introduction to Algebra",
                                "duration": 0.5,
                                "url": "https://example.com/algebra-intro"
                            },
                            {
                                "title": "Solving Linear Equations",
                                "duration": 0.75,
                                "url": "https://example.com/linear-equations"
                            }
                        ],
                        "quiz": [
                            {
                                "title": "Basic Algebra Quiz",
                                "questions": 10,
                                "duration": 0.25,
                                "url": "https://example.com/algebra-quiz"
                            }
                        ],
                        "practice": [
                            {
                                "title": "Algebra Practice Problems",
                                "problems": 20,
                                "duration": 0.5,
                                "url": "https://example.com/algebra-practice"
                            }
                        ]
                    }
                },
                "geometry": {
                    "name": "Geometry",
                    "difficulty": 2,
                    "estimated_time": 1.5,
                    "prerequisites": ["algebra"],
                    "formats": ["reading", "quiz", "practice"],
                    "content": {
                        "reading": [
                            {
                                "title": "Geometry Fundamentals",
                                "duration": 0.5,
                                "url": "https://example.com/geometry-fundamentals"
                            }
                        ],
                        "quiz": [
                            {
                                "title": "Geometry Quiz",
                                "questions": 15,
                                "duration": 0.3,
                                "url": "https://example.com/geometry-quiz"
                            }
                        ],
                        "practice": [
                            {
                                "title": "Geometry Practice Problems",
                                "problems": 25,
                                "duration": 0.7,
                                "url": "https://example.com/geometry-practice"
                            }
                        ]
                    }
                },
                "calculus": {
                    "name": "Calculus",
                    "difficulty": 4,
                    "estimated_time": 3.0,
                    "prerequisites": ["algebra", "geometry"],
                    "formats": ["video", "reading", "practice"],
                    "content": {
                        "video": [
                            {
                                "title": "Introduction to Calculus",
                                "duration": 1.0,
                                "url": "https://example.com/calculus-intro"
                            }
                        ],
                        "reading": [
                            {
                                "title": "Calculus Concepts",
                                "duration": 0.75,
                                "url": "https://example.com/calculus-concepts"
                            }
                        ],
                        "practice": [
                            {
                                "title": "Calculus Practice Problems",
                                "problems": 30,
                                "duration": 1.25,
                                "url": "https://example.com/calculus-practice"
                            }
                        ]
                    }
                }
            }
        },
        "physics": {
            "name": "Physics",
            "topics": {
                "mechanics": {
                    "name": "Mechanics",
                    "difficulty": 3,
                    "estimated_time": 2.5,
                    "prerequisites": ["algebra", "geometry"],
                    "formats": ["video", "quiz", "practice"],
                    "content": {
                        "video": [
                            {
                                "title": "Introduction to Mechanics",
                                "duration": 0.75,
                                "url": "https://example.com/mechanics-intro"
                            }
                        ],
                        "quiz": [
                            {
                                "title": "Mechanics Quiz",
                                "questions": 20,
                                "duration": 0.4,
                                "url": "https://example.com/mechanics-quiz"
                            }
                        ],
                        "practice": [
                            {
                                "title": "Mechanics Practice Problems",
                                "problems": 25,
                                "duration": 1.35,
                                "url": "https://example.com/mechanics-practice"
                            }
                        ]
                    }
                }
            }
        }
    }
}

def save_dataset():
    """Save the dataset to a JSON file"""
    os.makedirs("data", exist_ok=True)
    with open("data/study_dataset.json", "w") as f:
        json.dump(STUDY_DATASET, f, indent=2)

def load_dataset():
    """Load the dataset from JSON file"""
    try:
        with open("data/study_dataset.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        save_dataset()
        return STUDY_DATASET

def get_topic_info(subject, topic):
    """Get detailed information about a specific topic"""
    dataset = load_dataset()
    try:
        return dataset["subjects"][subject]["topics"][topic]
    except KeyError:
        return None

def get_available_topics(subject):
    """Get all available topics for a subject"""
    dataset = load_dataset()
    try:
        return list(dataset["subjects"][subject]["topics"].keys())
    except KeyError:
        return []

def get_available_subjects():
    """Get all available subjects"""
    dataset = load_dataset()
    return list(dataset["subjects"].keys())

if __name__ == "__main__":
    save_dataset()
    print("Dataset saved to data/study_dataset.json") 