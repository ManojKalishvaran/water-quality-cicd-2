import os
import joblib 
import pandas as pd
from flask import Flask, request, jsonify


# def init():
#     global model 
#     model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), "model.pkl")
#     model = joblib.load(model_path)


# def raw(raw_data):
#     data = json.loads(raw_data)['data']
#     data = pd.DataFrame(data)
#     print(data)
#     prediction = model.predict(data)
#     return prediction.tolist()


# if __name__ == "__main__":
#     init()  
#     with open(
#         r"D:\Azure MLOps learning\Custom project\water quality\Told\Water Quality Prediction\samples.json",
#         'r'
#     ) as f:
#         sample_json = f.read()
#         print("Prediction:", raw(sample_json))


app = Flask(__name__)

# Load model once when the container starts
model_path = os.getenv("MODEL_PATH", "./model_files/model.pkl")
model = joblib.load(model_path)

@app.route("/score", methods=["POST"])
def score():
    try:
        # Expect JSON input with features, e.g. {"data": [[5.1, 3.5, 1.4, 0.2]]}
        input_json = request.get_json(force=True)
        data = input_json.get("data")

        # Validate input
        if data is None:
            return jsonify({"error": "No data provided"}), 400

        data = pd.DataFrame(data)
        # Run prediction
        predictions = model.predict(data)

        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    # Run the Flask app on port 5001 (matching Dockerfile EXPOSE)
    app.run(host="0.0.0.0", port=5001)