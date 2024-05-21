from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load and preprocess the data
try:
    data = pd.read_csv("realfishdataset.csv")
    x_data = data.drop(columns=['fish'])
    z_scores = np.abs((x_data - x_data.mean()) / x_data.std())
    threshold = 3
    outliers = z_scores > threshold
    cleaned_df = x_data[~outliers.any(axis=1)]
    cleaned_dependent_column = data.loc[~outliers.any(axis=1), 'fish']
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(cleaned_df)
    X_train, X_test, y_train, y_test = train_test_split(normalized_data, cleaned_dependent_column, test_size=0.2, random_state=42)

    # Train the models
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2)
    rf_model.fit(X_train, y_train)
except Exception as e:
    print(f"Error during data preprocessing or model training: {e}")

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.json['features']
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)

        # Predict using Random Forest model
        prediction = rf_model.predict(features)[0]

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
