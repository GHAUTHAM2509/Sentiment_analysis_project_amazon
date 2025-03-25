from datasets import load_dataset
from app import predict_rating
import pandas as pd
import numpy as np

# ------------------------------------
# LOAD DATASET
# ------------------------------------
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Health_and_Personal_Care", trust_remote_code=True)

# Extract the "full" split
data = dataset["full"][:100000]
ratings = dataset["full"][:100000]["rating"]
reviews = dataset["full"][:100000]["text"]

# ------------------------------------
# EVALUATION VARIABLES
# ------------------------------------
correct_positive = 0
correct_negative = 0
incorrect_positive = 0
incorrect_negative = 0

# ------------------------------------
# REVIEW EVALUATION LOOP
# ------------------------------------
for i in range(100000):
    review = reviews[i]     # Accessing the 'text' field correctly
    rating = ratings[i]    # Accessing the 'rating' field correctly

    prediction = predict_rating(review)

    # Update counters based on prediction correctness
    if prediction == "positive" and rating > 3:
        correct_positive += 1
    elif prediction == "negative" and rating <= 3:
        correct_negative += 1
    elif prediction == "positive" and rating <= 3:
        incorrect_positive += 1
    elif prediction == "negative" and rating > 3:
        incorrect_negative += 1

# ------------------------------------
# RESULTS
# ------------------------------------
print(f"Correct positive: {correct_positive}")
print(f"Correct negative: {correct_negative}")
print(f"Incorrect positive: {incorrect_positive}")
print(f"Incorrect negative: {incorrect_negative}")