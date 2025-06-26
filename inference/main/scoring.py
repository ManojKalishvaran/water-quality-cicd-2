import pandas as pd
from flask import Flask, request, jsonify
from model_loader import load

app = Flask(__name__)
print("loading model...")
model = load()  

print("finished loading model...")
@app.route("/")
def home():
    return {"status":"okay"}


@app.route("/predict", methods=["POST"])
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
