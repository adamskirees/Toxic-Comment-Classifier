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
#%% [11] NEURAL NETWORK SENSITIVITY TEST
import numpy as np
from sklearn.metrics import classification_report

# 1. Get the raw probabilities from the NN
# These are the decimals between 0 and 1
nn_probs = nn_model.predict(X_test_nn).flatten()

# 2. Test a range of thresholds
thresholds = [0.2, 0.3, 0.4, 0.5]

for t in thresholds:
    print(f"\n" + "="*40)
    print(f"RESULTS FOR THRESHOLD: {t}")
    print("="*40)
    
    # Apply the threshold
    t_preds = (nn_probs >= t).astype(int)
    
    # Print the report for the Toxic Class (1)
    report = classification_report(y_test_nn, t_preds, output_dict=True)
    toxic_stats = report['1']
    
    print(f"Precision: {toxic_stats['precision']:.2f}")
    print(f"Recall:    {toxic_stats['recall']:.2f}")
    print(f"F1-Score:  {toxic_stats['f1-score']:.2f}")
# %%
#%% [12] TRAIN WORD2VEC EMBEDDINGS
from gensim.models import Word2Vec

print("\nStep 12: Training Word2Vec... learning word meanings.")

# 1. Prepare sentences (Word2Vec expects a list of lists of words)
sentences = [text.split() for text in df['clean_text']]

# 2. Train the Word2Vec model
# vector_size=100 means each word is represented by 100 numbers
# window=5 looks at 5 words before and after
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)

# 3. Test the "Brain" - Let's see what the model thinks is similar to 'hate'
try:
    similar_words = w2v_model.wv.most_similar('hate', topn=5)
    print("\nWords most similar to 'hate':")
    for word, score in similar_words:
        print(f"- {word}: {score:.2f}")
except KeyError:
    print("'hate' not in vocabulary.")

# Save the W2V brain
w2v_model.save("models/toxic_w2v.model")
# %%
#%% [13] CREATING THE EMBEDDING MATRIX
print("\nStep 13: Building the Embedding Matrix for the Neural Network...")

# 1. Create a matrix of zeros
embedding_matrix = np.zeros((vocab_size, 100))

# 2. Fill the matrix with our Word2Vec values
for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]

print(f"Matrix created. Shape: {embedding_matrix.shape}")

#%% [14] TRAINING THE W2V-POWERED NEURAL NETWORK
from tensorflow.keras.initializers import Constant

# Build a new model using the pre-trained weights
w2v_nn_model = Sequential([
    # We lock the weights initially so the model doesn't "forget" what it learned in W2V
    Embedding(vocab_size, 100, 
              embeddings_initializer=Constant(embedding_matrix), 
              input_length=max_length, 
              trainable=False),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

w2v_nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\nTraining the W2V-powered model...")
w2v_history = w2v_nn_model.fit(
    X_train_nn, y_train_nn, 
    epochs=10, 
    validation_data=(X_test_nn, y_test_nn),
    verbose=1
)
# %%
#%% [15] THE FINAL SHOWDOWN: BASELINE vs. NN vs. W2V-NN
from sklearn.metrics import f1_score, precision_score, recall_score

# 1. Get Probabilities for all models
# Baseline (Logistic)
baseline_probs = model.predict_proba(X_test)[:, 1]
# Standard NN
nn_probs = nn_model.predict(X_test_nn).flatten()
# Word2Vec NN
w2v_probs = w2v_nn_model.predict(X_test_nn).flatten()

# 2. Apply the winning thresholds
t_logic = 0.7  # Our tuned baseline
t_nn = 0.2     # Our tuned NN threshold

preds_baseline = (baseline_probs >= t_logic).astype(int)
preds_nn = (nn_probs >= t_nn).astype(int)
preds_w2v = (w2v_probs >= t_nn).astype(int)

# 3. Create a Comparison Table
results = {
    "Model": ["Logistic (Baseline)", "Standard NN", "Word2Vec + NN"],
    "Precision": [
        precision_score(y_test, preds_baseline),
        precision_score(y_test_nn, preds_nn),
        precision_score(y_test_nn, preds_w2v)
    ],
    "Recall": [
        recall_score(y_test, preds_baseline),
        recall_score(y_test_nn, preds_nn),
        recall_score(y_test_nn, preds_w2v)
    ],
    "F1-Score": [
        f1_score(y_test, preds_baseline),
        f1_score(y_test_nn, preds_nn),
        f1_score(y_test_nn, preds_w2v)
    ]
}

df_results = pd.DataFrame(results)
print("\n--- FINAL MODEL COMPARISON ---")
print(df_results.to_string(index=False))

# Calculate the winner
winner = df_results.iloc[df_results['F1-Score'].idxmax()]['Model']
print(f"\n🏆 THE CHAMPION IS: {winner}")
# %%
# Save your champion model one last time
nn_model.save('models/toxic_nn_champion.keras')

# Save the tokenizer (essential for deployment!)
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Champion model and tokenizer locked and loaded!")
# %%
