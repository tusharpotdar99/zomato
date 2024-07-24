import re
import demoji
from nltk.tokenize import word_tokenize
from string import punctuation
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Zomato:
    def __init__(self, review, tokenizer, model_w2v):
        self.review = review
        self.tokenizer = tokenizer
        self.model_w2v = model_w2v
        self.contractions = {
            "ain't": "are not", "aren't": "are not", "can't": "cannot", "could've": "could have", "couldn't": "could not",
            "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he had or he would", "he'll": "he will or he shall", "he's": "he is or he has",
            "how'd": "how did or how would", "how'll": "how will", "how's": "how is or how does", "I'd": "I had or I would",
            "I'll": "I will", "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it would", "it'll": "it will",
            "it's": "it is", "let's": "let us", "might've": "might have", "must've": "must have", "mustn't": "must not",
            "needn't": "need not", "shan't": "shall not", "she'd": "she had or she would", "she'll": "she will",
            "she's": "she is or she has", "should've": "should have", "shouldn't": "should not", "that'd": "that would",
            "that's": "that is", "there'd": "there would", "there'll": "there will", "there're": "there are", "there's": "there is",
            "they'd": "they had or they would", "they'll": "they will", "they're": "they are", "they've": "they have",
            "wasn't": "was not", "we'd": "we had or we would", "we'll": "we will", "we're": "we are", "we've": "we have",
            "weren't": "were not", "what'd": "what did or what would", "what'll": "what will", "what're": "what are",
            "what's": "what is", "what've": "what have", "when's": "when is", "where'd": "where did or where would",
            "where'll": "where will", "where're": "where are", "where's": "where is", "where've": "where have",
            "who'd": "who had or who would", "who'll": "who will", "who're": "who are", "who's": "who is", "who've": "who have",
            "why'd": "why did or why would", "why'll": "why will", "why're": "why are", "why's": "why is", "won't": "will not",
            "would've": "would have", "wouldn't": "would not", "you'd": "you had or you would", "you'll": "you will",
            "you're": "you are", "you've": "you have"
        }
        self.preprocessed_review = self.preprocess_text(review)
        self.sentiment, self.probability = self.predict_sentiment()

    def replace_contractions(self, sentence):
        words = sentence.split()
        new_sentence = []
        for word in words:
            if word.lower() in self.contractions:
                new_sentence.append(self.contractions[word.lower()])
            else:
                new_sentence.append(word)
        return ' '.join(new_sentence)

    def tokenize(self, text):
        clean_text = word_tokenize(text)
        return clean_text

    def remove_num(self, text):
        new_sentence = re.sub(r'\d+', '', text)
        return new_sentence

    def remove_punctuations(self, text):
        cleaned_text = []
        for word in text.split():
            if not any(char in punctuation for char in word):
                cleaned_text.append(word)
        return ' '.join(cleaned_text)

    def preprocess_text(self, review):
        review = review.lower()
        review = demoji.replace_with_desc(review)
        review = self.replace_contractions(review)
        review = self.remove_num(review)
        review = self.remove_punctuations(review)
        review = ' '.join(self.tokenize(review))
        return review

    def predict_sentiment(self):
        review = self.preprocess_text(self.review)
        review_sequence = self.tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(review_sequence, maxlen=80, padding='pre')
        prediction = self.model_w2v.predict(padded_sequence)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        probability = prediction[0][0] if sentiment == 'Positive' else 1 - prediction[0][0]
        return sentiment, probability