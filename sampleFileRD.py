import json
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib
# Specify the path to your JSON file
json_file_path = '/Users/saiharshinigarapati/Desktop/model_params.json'

# Load the JSON data from the file
with open(json_file_path, 'r') as json_file:
    model_params = json.load(json_file)

num_trees = model_params['NumTrees']
num_var_to_sample = model_params['NumVarToSample']
loaded_model = joblib.load('/Users/saiharshinigarapati/Desktop/trained_rf_model1.pkl')
# Create the model
rf_model = RandomForestRegressor(n_estimators=num_trees, max_features=num_var_to_sample)
# Sample new data for prediction (replace this with your actual data)
new_data = np.array([[1272, 836, 1205, 1490, 1110, 11.2, 59.6, 0.7888]])

# Make predictions on new data
predictions = rf_model.predict(new_data)

# Print the predictions
print("Predictions:", predictions)
