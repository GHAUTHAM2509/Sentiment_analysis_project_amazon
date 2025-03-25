from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import project1 as p1
import pandas as pd
import io
import json
import os

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
    return render_template('index.html')

@app.route('/csv-analysis')
def csv_analysis():
    return render_template('csv_analysis.html')

@app.route('/accuracy-analysis')
def accuracy_analysis():
    return render_template('accuracy_analysis.html')

@app.route('/get-accuracy-data')
def get_accuracy_data():
    try:
        with open('static/accuracy_data.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']
    prediction = predict_rating(review)
    return jsonify({'prediction': prediction})

@app.route('/analyze-csv', methods=['POST'])
def analyze_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'}), 400
    
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Get first 100 reviews
        reviews = df['review'].head(100).tolist()
        
        # Analyze each review
        positive_count = 0
        negative_count = 0
        
        for review in reviews:
            prediction = predict_rating(review)
            if prediction == "positive":
                positive_count += 1
            else:
                negative_count += 1
        
        return jsonify({
            'positive_count': positive_count,
            'negative_count': negative_count,
            'total_reviews': len(reviews),
            'total_available': len(df['review'])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    # Set host to '0.0.0.0' to make it accessible from other machines
    app.run(host='0.0.0.0', port=5001, debug=True)