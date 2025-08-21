from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from transformers import BertTokenizer

app = Flask(__name__)
CORS(app)

# Load tokenizer dan model
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
saved_model_path = "saved_model"  # Ganti sesuai folder modelmu
tfsmlayer = tf.saved_model.load(saved_model_path).signatures["serving_default"]

@app.route('/')
def home():
    return render_template('index.html') 

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data["text"]

        # Tokenisasi input
        tokens = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="np")
        input_ids = tf.convert_to_tensor(tokens["input_ids"], dtype=tf.int32)
        attention_mask = tf.convert_to_tensor(tokens["attention_mask"], dtype=tf.int32)
        token_type_ids = tf.convert_to_tensor(tokens["token_type_ids"], dtype=tf.int32)

        # Panggil model
        outputs = tfsmlayer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Ambil hasil prediksi
        logits = outputs["logits"].numpy()
        label_id = np.argmax(logits, axis=1)[0]
        label = "positif" if label_id == 1 else "negatif"

        return jsonify({"label": label})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
