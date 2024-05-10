from flask import Flask, request, jsonify, render_template
from logic import prepare_data_and_train, predict_next_word
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
import os

app = Flask(__name__)

# Paths to the model and tokenizer
model_path = "my_model.h5"
tokenizer_path = "tokenizer.json"

# Load or train the model and tokenizer
if os.path.exists(model_path) and os.path.exists(tokenizer_path):
    # Load the model
    model = load_model(model_path)

    # Load the tokenizer from JSON
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

    # Set your max_len (this should match the model's input length)
    max_len = 18  # Ensure this value aligns with your model training
else:
    # If files don't exist, train the model and save it
    model, tokenizer, max_len = prepare_data_and_train("data.txt")
    model.save(model_path)

    # Save tokenizer to JSON
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Input text cannot be empty"}), 400

        predicted_word = predict_next_word(model, tokenizer, text, max_len)

        if predicted_word is None:
            return jsonify({"error": "Could not determine predicted word"}), 400

        return jsonify({"predicted_word": predicted_word})

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500
