from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)

# Train and save a simple IRIS model
def train_model():
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(iris.data, iris.target)
    joblib.dump(model, 'model.joblib')
    return model

# Load model
if os.path.exists('model.joblib'):
    model = joblib.load('model.joblib')
else:
    model = train_model()

# Load iris data for feature names
iris = load_iris()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Handle both array format and dict format
        if 'features' in data:
            # Array format: [5.1, 3.5, 1.4, 0.2]
            features = np.array(data['features']).reshape(1, -1)
        else:
            # Dict format with feature names
            features = np.array([[
                data.get('sepal_length', 0),
                data.get('sepal_width', 0),
                data.get('petal_length', 0),
                data.get('petal_width', 0)
            ]])
        
        # Make prediction
        prediction = model.predict(features)
        species = iris.target_names[prediction[0]]
        
        return jsonify({
            'prediction': int(prediction[0]),
            'species': species,
            'features_received': features.tolist()[0]
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)



# from flask import Flask, request, jsonify
# import pickle
# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier
# import os

# app = Flask(__name__)

# # Train and save a simple IRIS model
# def train_model():
#     iris = load_iris()
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(iris.data, iris.target)
#     with open('model.pkl', 'wb') as f:
#         pickle.dump(model, f)
#     return model

# # Load model
# import joblib  # Add this line

# # ...

# if os.path.exists('model.joblib'):
#     model = joblib.load('model.joblib')  # Use joblib.load
# else:
#     model = train_model()
#     joblib.dump(model, 'model.joblib')  # Save using joblib.dump


# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({'status': 'healthy'}), 200

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         features = np.array(data['features']).reshape(1, -1)
#         prediction = model.predict(features)
#         iris = load_iris()
#         species = iris.target_names[prediction[0]]
        
#         return jsonify({
#             'prediction': int(prediction[0]),
#             'species': species
#         }), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)
