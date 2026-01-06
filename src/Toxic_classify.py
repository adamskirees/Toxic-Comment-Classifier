# Import Libraries
#%% [1] IMPORT LIBRARIES
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
nltk.download(['stopwords', 'wordnet', 'omw-1.4', 'punkt'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize these GLOBALLY so every cell can see them
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#%% [2] DATA LOADING & PREPROCESSING
# Bring in Raw data and create classification column
#csv_path = '../data/Toxic.csv'

# 1. Get the path of the script you are currently in (src/Toxic_classify.py)
script_path = os.path.abspath(__file__)

# 2. Get the folder the script is in (src)
script_dir = os.path.dirname(script_path)

# 3. Get the folder ABOVE that (The project root)
root_dir = os.path.dirname(script_dir)

# 4. Join that root path with the data folder
csv_path = os.path.join(root_dir, 'data', 'Toxic.csv')

print(f"The raw data is here -: {csv_path}")

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

#%% [3] Functions  
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

#%% [4] Run Preprocessing 
# Run the preprocessing function
print("Applying preprocessing... this may take a moment.")
df['clean_text'] = df['comment_text'].apply(preprocess_text)
print(df[['comment_text', 'clean_text']].head())

#%% [5] VECTORIZATION with TF-IDF and Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

print("\nStep 5: Converting text to numbers (Vectorization)...")

# TF-IDF (Term Frequency-Inverse Document Frequency)
# It rewards words that are rare but meaningful (like specific insults)
# and ignores words that are common but useless (like 'the', 'and').
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# X is our matrix of numbers (features)
# y is our target (is_toxic)
X = tfidf.fit_transform(df['clean_text'])
y = df['is_toxic']

print(f"Feature Matrix Shape: {X.shape}")

#%% [6] BALANCED MODEL TRAINING & EVALUATION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

print("\nStep 6: Training the Balanced Logistic Regression Model...")

# 1. Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize Model with 'balanced' logic
# 'class_weight=balanced' tells the AI to penalize missing a Toxic comment 
# much more heavily than missing a Clean one.
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

# 3. Train
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Output Results
print("\n--- MODEL PERFORMANCE REPORT ---")
print(classification_report(y_test, y_pred))

# ROC-AUC is the best metric for imbalanced data
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"ROC-AUC Score: {roc_auc:.4f}")
# %%
#%% [7] LIVE TESTING & THRESHOLDING
# Get probabilities instead of hard 0 or 1
y_probs = model.predict_proba(X_test)[:, 1]

# Set a stricter threshold (0.7 instead of 0.5)
custom_threshold = 0.7
y_pred_stricter = (y_probs >= custom_threshold).astype(int)

print(f"--- RESULTS WITH {custom_threshold} THRESHOLD ---")
print(classification_report(y_test, y_pred_stricter))

# LIVE TEST FUNCTION
def test_comment(text):
    clean = preprocess_text(text)
    vec = tfidf.transform([clean])
    prob = model.predict_proba(vec)[0][1]
    return f"Comment: {text}\n- Toxic Probability: {prob:.2%}"

print(test_comment("You are a total idiot and I hate you"))
print(test_comment("I disagree with your point, but let's talk more."))

# %%
#%% [8] TENSORFLOW NEURAL NETWORK
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("\nStep 8: Preparing Neural Network...")

# 1. Hyperparameters
vocab_size = 10000 
max_length = 100
embedding_dim = 16

# 2. Tokenize for Deep Learning (Different from TF-IDF)
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# 3. Train/Test Split for NN
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(padded, y, test_size=0.2, random_state=42)

# 4. Build the Model
nn_model = Sequential([
    # Embedding layer learns the relationships between words
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dropout(0.2), # Prevents overfitting
    Dense(1, activation='sigmoid') # Binary output (0 to 1)
])

nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\nTraining Neural Network... this might take 1-2 minutes.")
history = nn_model.fit(
    X_train_nn, y_train_nn, 
    epochs=10, 
    validation_data=(X_test_nn, y_test_nn),
    verbose=1
)
# %%
#%% [9] COMPARE NEURAL NETWORK VS BASELINE
from sklearn.metrics import classification_report

# 1. Get predictions from the Neural Network (probabilities)
nn_probs = nn_model.predict(X_test_nn)

# 2. Convert probabilities to hard classes (using 0.7 to match our tuned baseline)
nn_preds = (nn_probs >= 0.7).astype(int)

print("\n--- NEURAL NETWORK RESULTS (Threshold 0.7) ---")
print(classification_report(y_test_nn, nn_preds))

print("\n--- REMINDER: BASELINE LOGISTIC REGRESSION (Threshold 0.7) ---")
# This prints the previous results for easy scrolling comparison
print(classification_report(y_test, y_pred_stricter))
# %%
#%% [10] SAVE MODELS
import pickle
import os

# Create the 'models' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')
    print("Created 'models' directory.")

# Save the Logistic Baseline & TF-IDF
with open('models/logistic_baseline.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Save the Neural Network
# Using .keras extension is the modern standard for TensorFlow 3+
nn_model.save('models/toxic_nn_model.keras')

print("All models saved successfully to E:/My Documents/CODING PROJECTS/Toxic-Comment-Classifier/models/")
# %%
