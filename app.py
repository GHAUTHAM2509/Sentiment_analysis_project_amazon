from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import project1 as p1
import pandas as pd
import io

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
        
        # Assuming the review text is in a column named 'review'
        # Modify this according to your CSV structure
        reviews = df['review'].tolist()
        
        # Analyze each review
        positive_count = 0
        negative_count = 0
        
        for review in reviews:
            prediction = predict_rating(str(review))
            if prediction == "positive":
                positive_count += 1
            else:
                negative_count += 1
        
        return jsonify({
            'positive_count': positive_count,
            'negative_count': negative_count,
            'total_reviews': len(reviews)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)