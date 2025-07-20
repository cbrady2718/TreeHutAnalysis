import logging
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def preprocess_text(text):
    """Clean and preprocess text for analysis."""
    try:
        if not isinstance(text, str):
            return ''
        text = text.lower()
        text = re.sub(r'[^\\w\\\s]', '', text)
        #treehut and skin seem like they don't add value here. 
        stop_words = set(stopwords.words('english') + ['treehut', 'skin'])
        tokens = word_tokenize(text)
        tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        logging.error(f"Error preprocessing text: {e}")
        return ''
    