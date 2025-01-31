from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import project1 as p1

app = Flask(__name__)

# Load the saved models
model_p = joblib.load("perceptron_model.pkl")
model_avg = joblib.load("average_perceptron_model.pkl")
model_peg = joblib.load("pegasos_model.pkl")
models = [model_p, model_avg, model_peg]

def predict_rating(review):
    prediction = 0
    for i, model in enumerate(models, start=1):
        theta = model["theta"]
        theta_0 = model["theta_0"]
        dictionary = model["dictionary"]
        
        # Extract features
        review_features = p1.extract_bow_feature_vectors([review], dictionary)
        
        # Predict rating
        prediction += p1.classify_star_rating(review_features, theta, theta_0)
    
    if prediction > 0:
        return "positive"
    else:
        return "negative"


@app.route('/')
def home():
    return render_template('index.html')  # Load HTML page

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from AJAX request
    review = data['review']  # Extract text input
    prediction = predict_rating(review)  # Make prediction
    return jsonify({'prediction': prediction})  # Return JSON response

if __name__ == "__main__":
    app.run(debug=True)