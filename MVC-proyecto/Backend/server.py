from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from model import NaiveBayes
from preprocess import clean_tweet
import pandas as pd
import os

# Configuración de rutas ABSOLUTAS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '../Frontend')

app = Flask(__name__,
            template_folder=os.path.join(FRONTEND_DIR, 'templates'),
            static_folder=os.path.join(FRONTEND_DIR, 'static'))
CORS(app)

# Dataset y modelo
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'tweets.csv')
model = NaiveBayes()

try:
    print("\nCargando dataset...")
    texts, labels = model.load_data(DATASET_PATH)
    print(f"Distribución inicial: {pd.Series(labels).value_counts().to_dict()}")
    
    print(" Entrenando modelo...")
    model.train([clean_tweet(text) for text in texts], labels)
    print(f"Modelo entrenado con {len(texts)} tweets\n")
except Exception as e:
    print(f"Error: {str(e)}")
    print("Usando datos de ejemplo mínimos")
    model.train([["happy"], ["sad"]], ["positive", "negative"])

# Rutas
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'tweet' not in data:
            return jsonify({"error": "Bad request"}, 400)
        
        cleaned = clean_tweet(data['tweet'])
        prediction = model.predict(cleaned)
        
        return jsonify({
            "sentiment": prediction,
            "tokens": cleaned
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)