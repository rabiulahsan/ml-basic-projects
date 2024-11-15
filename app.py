from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.pipelines.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for a home page
@app.route('/')
def index():
    return jsonify({"message": "Welcome to the ML Prediction API!"})

# Route for making predictions
@app.route('/predictdata', methods=['POST'])
def predictdata():
    try:
        #log the route hit
        print("hitting the route successfully...")  
        # Retrieve data from the request JSON
        data = request.json
        
        # Create a CustomData instance with the input data
        custom_data = CustomData(
            gender=data.get('gender'),
            race_ethnicity=data.get('ethnicity'),
            parental_level_of_education=data.get('parental_level_of_education'),
            lunch=data.get('lunch'),
            test_preparation_course=data.get('test_preparation_course'),
            reading_score=float(data.get('reading_score')),
            writing_score=float(data.get('writing_score'))
        )
        
        # Convert the input data to a DataFrame
        pred_df = custom_data.get_data_as_data_frame()
        
        # Initialize the prediction pipeline and make a prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Return the result as a JSON response
        return jsonify({"prediction": results[0]})

    except Exception as e:
        raise CustomException(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
