# Import Libraries
import pandas as pd
import numpy as np
import re
import io
import string
import shutil
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time  # For measuring training time

from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.decomposition import PCA

# -- NLTK Setup --
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Bring in Raw data and create classification column
csv_path = 'data/Toxic.csv'

df = pd.read_csv(csv_path)
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("Success: Data loaded from local folder!")
else:
    print(f"Error: Could not find file at {os.path.abspath(csv_path)}")

# --- Labeling ---
toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# Create a 1 if ANY toxic column is 1, else 0
df['is_toxic'] = df[toxicity_columns].any(axis=1).astype(int)

# Check the balance of your classes
print(df['is_toxic'].value_counts(normalize=True))


################################## FUNCTIONS ################ FUNCTIONS #####################
### FUNCTION 1.  Text Preprocessing: Cleaning, Tokenizing, Lemmatizing #####################
def preprocess_text(text):
    #1. Handle potential empty values from CSV
    if not isinstance(text, str): 
        return ""

# 2. Lowercase and strip whitespace
    text = text.lower().strip()
    
    # 3. Remove punctuation and numbers in one pass using regex
    # [^\w\s] removes punctuation, \d+ removes numbers
    text = re.sub(r'[^\w\s]|\d+', '', text)
    
    # 4. Tokenize and clean in a single list comprehension
    # thought about word length (e.g., between 2 and 20 chars) - dont wnat long meaningless  words
    words = [
        lemmatizer.lemmatize(word) 
        for word in text.split() 
        if word not in stop_words and 2 < len(word) < 20
    ]

    # 5. Join back into a string (Standard for Vectorizers)
    return ' '.join(words)
############################################################ End Function 1. ###############

# Run the preprocessing function
print("Applying preprocessing... this may take a moment.")
df['clean_text'] = df['comment_text'].apply(preprocess_text)
print(df[['comment_text', 'clean_text']].head())
