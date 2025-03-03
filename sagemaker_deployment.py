# sagemaker_deployment.py
import boto3
import sagemaker
import os
import sys

def deploy_to_sagemaker():
    print("Note: This script simulates SageMaker deployment steps.")
    print("To actually deploy to AWS, you would need AWS credentials configured.")
    
    # Create model directory structure
    os.makedirs('model/code', exist_ok=True)
    
    # Write inference code
    with open('model/code/inference.py', 'w') as f:
        f.write('''
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
''')
    
    print("\nCreated inference.py script at model/code/inference.py")
    
    # Create a tar.gz file (mock operation for demo)
    print("In a real deployment, you would create a model.tar.gz file with:")
    print("tar -czf model.tar.gz -C model .")
    
    # Simulate S3 upload
    print("\nYou would then upload the model to S3:")
    print("model_data = sagemaker_session.upload_data('model.tar.gz', bucket=bucket, key_prefix=prefix)")
    
    # Simulate SageMaker model creation
    print("\nCreate SageMaker model:")
    print("""
sklearn_model = SKLearnModel(
    model_data=model_data,
    role=role,
    entry_point='inference.py',
    framework_version='0.23-1'
)
    """)
    
    # Simulate deployment
    print("\nDeploy to endpoint:")
    print("""
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)
    """)
    
    # Simulate test
    print("\nTest the endpoint:")
    print("""
test_data = {
    'userId': 1,
    'movieIds': [1, 2, 3, 4, 5]
}
response = predictor.predict(test_data)
print(response)
    """)
    
    print("\nSageMaker deployment simulation complete!")
    print("In a real scenario, you would need AWS credentials and incur AWS charges.")

if __name__ == "__main__":
    deploy_to_sagemaker()