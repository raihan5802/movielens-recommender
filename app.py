# app.py
from flask import Flask, request, jsonify
import pickle
import os
import pandas as pd
import traceback

app = Flask(__name__)

# Load models with error handling
try:
    print("Loading models...")
    user_based = pickle.load(open('model/user_based_model.pkl', 'rb'))
    item_based = pickle.load(open('model/item_based_model.pkl', 'rb'))
    svd = pickle.load(open('model/svd_model.pkl', 'rb'))
    movies = pd.read_csv('ml-latest-small/movies.csv')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    print(traceback.format_exc())
    raise e

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'active',
        'message': 'MovieLens Recommender API is running',
        'endpoints': {
            '/predict': 'POST - Get a prediction for a user-movie pair',
            '/recommend/user': 'GET - Get recommendations for a user'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    print(f"Received predict request: {request.data}")
    try:
        data = request.get_json()
        print(f"Parsed JSON data: {data}")
        
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        user_id = data.get('userId')
        movie_id = data.get('movieId')
        model_type = data.get('model', 'svd')  # Default to SVD
        
        if not user_id or not movie_id:
            return jsonify({'error': 'userId and movieId are required'}), 400
        
        # Convert to proper types in case they come as strings
        user_id = int(user_id)
        movie_id = int(movie_id)
        
        print(f"Making prediction for user {user_id}, movie {movie_id} using {model_type} model")
        
        if model_type == 'user_based':
            prediction = user_based.predict(user_id, movie_id)
        elif model_type == 'item_based':
            prediction = item_based.predict(user_id, movie_id)
        else:
            prediction = svd.predict(user_id, movie_id)
        
        result = {
            'userId': user_id,
            'movieId': movie_id,
            'predictedRating': float(prediction.est),
            'model': model_type
        }
        print(f"Prediction result: {result}")
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/recommend/user', methods=['GET'])
def recommend_for_user():
    try:
        user_id = request.args.get('userId')
        if not user_id:
            return jsonify({'error': 'userId is required'}), 400
        
        user_id = int(user_id)
        print(f"Generating recommendations for user {user_id}")
        
        # Get all movies
        all_movie_ids = movies['movieId'].tolist()
        
        # Generate predictions for all movies
        predictions = []
        for movie_id in all_movie_ids[:100]:  # Limit to first 100 for demo purposes
            try:
                prediction = svd.predict(user_id, movie_id)
                predictions.append((movie_id, prediction.est))
            except Exception as e:
                print(f"Error predicting movie {movie_id}: {str(e)}")
                continue
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_movies = predictions[:10]
        
        # Get movie details
        recommendations = []
        for movie_id, rating in top_movies:
            movie = movies[movies['movieId'] == movie_id]
            if not movie.empty:
                recommendations.append({
                    'movieId': int(movie_id),
                    'title': movie['title'].values[0],
                    'genres': movie['genres'].values[0],
                    'predictedRating': float(rating)
                })
        
        result = {
            'userId': user_id,
            'recommendations': recommendations
        }
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in recommend endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Add an endpoint for debugging model files
@app.route('/debug', methods=['GET'])
def debug():
    try:
        model_files = os.listdir('model') if os.path.exists('model') else []
        data_files = os.listdir('ml-latest-small') if os.path.exists('ml-latest-small') else []
        
        return jsonify({
            'model_directory_exists': os.path.exists('model'),
            'model_files': model_files,
            'data_directory_exists': os.path.exists('ml-latest-small'),
            'data_files': data_files,
            'working_directory': os.getcwd()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    app.run(debug=True, port=5000, host='0.0.0.0')