from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load('movie_revenue_model.pkl')
feature_columns = joblib.load('model_features.pkl')

# All possible genres the model was trained on
GENRE_LIST = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
              'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']

@app.route('/')
def home():
    return "Movie Revenue Predictor API is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Base numeric features
        input_dict = {
            'budget': data['budget'],
            'runtime': data['runtime'],
            'release_year': data['release_year'],
            'release_month': data['release_month'],
            'num_production_companies': data.get('num_production_companies', 1),
            'num_production_countries': data.get('num_production_countries', 1),
            'num_spoken_languages': data.get('num_spoken_languages', 1),
        }

        # One-hot encode genres from a list, e.g. ["Action", "Adventure"]
        input_genres = data.get('genres', [])
        for genre in GENRE_LIST:
            input_dict[genre] = 1 if genre in input_genres else 0

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_columns]  # ensure correct column order

        pred_log = model.predict(input_df)[0]
        pred_revenue = float(np.expm1(pred_log))  # reverse the log transform

        return jsonify({'predicted_revenue': round(pred_revenue, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)