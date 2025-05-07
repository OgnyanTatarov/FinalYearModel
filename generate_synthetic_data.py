# generate_synthetic_data.py
import random
import json

# Simulated subject content
subject_db = [
    {"id": "math_algebra", "subject": "Math", "topic": "Algebra", "difficulty_level": 3, "estimated_time": 1.0, "study_format": ["video", "quiz"]},
    {"id": "math_geometry", "subject": "Math", "topic": "Geometry", "difficulty_level": 2, "estimated_time": 0.5, "study_format": ["reading", "quiz"]},
    {"id": "math_probability", "subject": "Math", "topic": "Probability", "difficulty_level": 4, "estimated_time": 1.5, "study_format": ["video", "reading"]}
]

# Generate random users and plans
def generate_random_data(num_users=100):
    all_rows = []
    for i in range(num_users):
        user_id = f"user_{i}"
        daily_time = round(random.uniform(0.5, 3.0), 1)
        preferred_formats = random.sample(["video", "reading", "quiz"], 2)
        days_left = random.randint(3, 14)
        subject = "Math"

        for topic in subject_db:
            confidence = random.randint(1, 5)
            performance = round(random.uniform(40, 95), 1)
            duration = round(random.uniform(0.5, min(topic["estimated_time"] + 1, daily_time)), 1)
            format_used = random.choice(topic["study_format"])

            row = {
                "subject": subject,
                "days_left": days_left,
                "daily_time": daily_time,
                "topic": topic["id"],
                "topic_difficulty": topic["difficulty_level"],
                "confidence": confidence,
                "performance": performance,
                "preferred_format_video": int("video" in preferred_formats),
                "preferred_format_reading": int("reading" in preferred_formats),
                "preferred_format_quiz": int("quiz" in preferred_formats),
                "duration": duration,
                "format": format_used
            }
            all_rows.append(row)
    return all_rows

if __name__ == "__main__":
    data = generate_random_data(300)
    with open("data/synthetic_training_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Synthetic dataset generated at: data/synthetic_training_data.json")
