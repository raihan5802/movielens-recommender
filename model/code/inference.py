
import pickle
import json
import numpy as np

def model_fn(model_dir):
    """Load the model for inference"""
    with open(f"{model_dir}/svd_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make prediction with the model"""
    user_id = input_data.get('userId')
    movie_ids = input_data.get('movieIds', [])
    
    if not movie_ids:
        return {"error": "No movieIds provided"}
    
    predictions = []
    for movie_id in movie_ids:
        try:
            pred = model.predict(user_id, movie_id)
            predictions.append({
                'movieId': movie_id,
                'predictedRating': pred.est,
                'details': {
                    'userId': pred.uid,
                    'movieId': pred.iid,
                    'actualRating': pred.r_ui if pred.r_ui is not None else "unknown"
                }
            })
        except:
            predictions.append({
                'movieId': movie_id,
                'error': 'Could not generate prediction'
            })
    
    return {"predictions": predictions}

def output_fn(prediction, response_content_type):
    """Format prediction output"""
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
