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
        prediction += p1.classify_star_rating(review_features, theta, theta_0, i)
    
    return int(np.ceil(prediction / 3))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review = request.form['review']
        rating = predict_rating(review)
        return jsonify({'rating': rating})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, port=5000)
