import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import demoji
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from nltk import word_tokenize
from flask import Flask, request, jsonify, render_template
from zomato import Zomato

app = Flask(__name__)

# Load models
wtv_model = Word2Vec.load('models/word2vec_model.bin')
tokenizer = pickle.load(open('models/tokenizer.pkl', 'rb'))
model = load_model('models/model_wtv.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    zomato_review = Zomato(review, tokenizer, model)
    predicted_sentiment = zomato_review.sentiment
    prediction_probability = zomato_review.probability
    return render_template('index.html', review=review, prediction=predicted_sentiment, probability=prediction_probability)

@app.route('/similar', methods=['POST'])
def similar():
    word = request.form['word']
    similar_words = []
    if word in wtv_model.wv:
        similar_words = wtv_model.wv.most_similar(word)
    return render_template('index.html', word=word, similar_words=similar_words)

if __name__ == '__main__':
    app.run(debug=True)
