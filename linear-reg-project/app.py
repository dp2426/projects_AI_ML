from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MODEL_FOLDER"] = MODEL_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB limit

MODEL_FILE = os.path.join(app.config["MODEL_FOLDER"], "model.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # Ensure this points to the correct template

@app.route("/train", methods=["POST"])
def train():
    file = request.files.get("file")  # Use .get() to avoid KeyError
    if file is None or file.filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Read the file based on extension
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file.filename.endswith(".xlsx"):
        df = pd.read_excel(file_path, engine="openpyxl")
    elif file.filename.endswith(".json"):
        df = pd.read_json(file_path)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    # Ensure 'x' and 'y' columns exist
    if "x" not in df.columns or "y" not in df.columns:
        return jsonify({"error": "File must contain 'x' and 'y' columns"}), 400

    # Train the model
    X = df["x"].values.reshape(-1, 1)
    y = df["y"].values

    model = LinearRegression()
    model.fit(X, y)

    # Save the model to a file
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved successfully!")  # Debugging log
    return jsonify({"message": "Model trained and saved successfully!"})

@app.route("/predict", methods=["POST"])
def predict():
    # Load the model from the file
    if not os.path.exists(MODEL_FILE):
        return jsonify({"error": "Model is not trained yet!"}), 400

    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)

        data = request.get_json()
        if "x" not in data:
            return jsonify({"error": "Missing input value 'x'"}), 400

        x_value = np.array([[float(data["x"])]])
        prediction = model.predict(x_value)[0]  # Extracting single value from array
        return jsonify({"prediction": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)