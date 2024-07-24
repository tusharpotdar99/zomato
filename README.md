# Zomato Review Sentiment Analysis

This project analyzes the sentiment of Zomato restaurant reviews using a machine learning model. The Flask-based web application takes user-submitted reviews, preprocesses the text, and predicts whether the sentiment is positive or negative. Additionally, the app provides the prediction probability and can display similar words using a pre-trained Word2Vec model.

## Features

- **Sentiment Analysis**: Predicts the sentiment (Positive/Negative) of user-submitted reviews along with the prediction probability.
- **Text Preprocessing**: Preprocesses text by lowercasing, removing numbers and punctuations, replacing contractions, and tokenizing.
- **Word2Vec Similar Words**: Displays similar words based on the pre-trained Word2Vec model.
- **User Experience**: Includes a loading spinner during the prediction process.

## Project Structure
Zomato-Review-Sentiment-Analysis/
│
├── app.py
├── models/
│ ├── word2vec_model.bin
│ ├── word2vec_model.bin.syn1neg.npy
│ ├── word2vec_model.bin.wv.vectors.npy
│ ├── tokenizer.pkl
│ └── model_wtv.h5
├── templates/
│ └── index.html
├── static/
│ └── css/
│ └── styles.css
├── requirements.txt
└── README.md

## Setup and Installation

### Prerequisites

- Python 3.6+
- Flask
- TensorFlow
- Gensim
- NLTK
- Demoji
- gdown

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Zomato-Review-Sentiment-Analysis.git
   cd Zomato-Review-Sentiment-Analysis

2. **Create a virtual environment:**
   python -m venv venv
   source venv/bin/activate

3. **install dependencies**
   pip install -r requirements.txt

4. **Download model files**
model_wtv.h5 ==  https://drive.google.com/file/d/1gslFmJ2wkhSl2NpiArtvkp07vzSIpaH1/view?usp=sharing
tokenizer.pkl == https://drive.google.com/file/d/1gWgN4ArmQV30DUoCLVbkVKA6dHlwyURU/view?usp=sharing
word2vec_model.bin == https://drive.google.com/file/d/1Mj9CtpsuzQbbombwOzqaOHZaDPTUAEf4/view?usp=sharing
word2vec_model.bin.syn1neg.npy ==    https://drive.google.com/file/d/1uKg0sye2iAE_EHUuMwAl33a2Mzg19pHq/view?usp=sharing
word2vec_model.bin.wv.vectors.npy == https://drive.google.com/file/d/1qTxgKI1zbD3hZcxmCbqwQ1mgfVoyZItC/view?usp=sharing


6. **Run the application:**

**Usage**
**Predict Sentiment:**
Enter a Zomato review in the text box and click on "Submit".
The predicted sentiment and its probability will be displayed.
If you want to find similar words, enter a word in the "Word2Vec Similar Words" section.


**License**
This project is licensed under the MIT License - see the LICENSE file for details.

**Acknowledgements**
The pre-trained Word2Vec model used in this project is based on the Gensim library.
Special thanks to the contributors of NLTK, TensorFlow, and Flask for their excellent tools and libraries.

