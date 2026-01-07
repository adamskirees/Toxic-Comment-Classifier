# Toxic Comment Classifier 🛡️

A Machine Learning project designed to identify and categorize toxic language in text data. This tool can detect multiple levels of toxicity, including threats, insults, and obscenity, making it useful for content moderation and online safety.

## 📌 Overview
Online platforms often struggle to maintain civil discourse. This project implements a Natural Language Processing (NLP) pipeline to automatically flag harmful comments. Using a multi-label classification approach, it identifies specific types of toxicity.

Use cases include managing social media comments or onine forum posts to maintiain/delete "toxic" comments. 

## The Goal & Methodology
The primary objective of this project is to build an NLP model that can generalize the logic of human moderation. Using a dataset where approximately 10.1% of comments are flagged as toxic (Ground Truth), we aim to move beyond simple keyword matching. By transforming raw text into a binary classification problem (Toxic vs. Clean), we are testing the model's ability to identify harmful patterns and subjective language. Success is measured not just by raw accuracy—which can be misleading due to the 89.9% class imbalance—but by the model's precision and recall in detecting actual toxicity in unseen data.

🚀 Business Value Proposition
This project demonstrates a production-focused approach to automated content moderation. In a high-volume commercial environment, manual review of user-generated content is both costly and slow. This model provides:

Operational Efficiency: Automates the initial screening of comments, allowing human moderators to focus only on "gray area" cases.

Brand Protection: Real-time detection of toxic, obscene, or threatening language to maintain a safe community and protect brand equity.

Scalable Compliance: Provides a standardized framework for content governance that scales without a linear increase in headcount.

📊 Performance Metrics
The model was evaluated using a multi-label approach to account for overlapping categories (e.g., a comment being both 'toxic' and 'obscene').

Primary Metric: [Insert your best metric here, e.g., F1-Score or AUC-ROC]

Key Feature: Implements [mention a specific technique you used, e.g., TF-IDF Vectorization or Class Weight Balancing] to handle imbalanced datasets typical of real-world spam/toxicity.

🛠️ Commercial Application
This classifier can be integrated via API into:

Customer Support Portals: To prioritize or flag aggressive tickets.

Internal Communication Tools: To ensure workplace policy compliance.

Marketing Dashboards: To filter sentiment analysis data for cleaner brand insights.



## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, Numpy, Tensorflow, Gensim, Scikit-learn, NLTK (Preprocessing)
- **Model:** [TBD: e.g., Logistic Regression / Random Forest / BERT]
- **Environment:** Virtualenv / VS Code

## 📁 Project Structure
```text
├── data/               # Raw and processed datasets (ignored by git)
├── notebooks/          # Exploratory Data Analysis (EDA)
├── src/                # Source code for preprocessing and training
├── venv/               # Virtual environment
├── .gitignore          # Files to exclude from version control
└── README.md           # Project documentation

🚀 Getting Started
Prerequisites
Python 3.8+

Kaggle Toxic Comment Dataset

Installation
Clone the repository:

git clone [https://github.com/adamskirees/Toxic-Comment-Classifier.git](https://github.com/adamskirees/Toxic-Comment-Classifier.git)
cd Toxic-Comment-Classifier
Set up virtual environment:

python -m venv venv
source venv/Scripts/activate  # Windows
Install dependencies:

pip install -r requirements.txt

📊 Roadmap

[x] Initial Project Setup

[ ] Exploratory Data Analysis (EDA)

[ ] Text Preprocessing & Tokenization

[ ] Model Training & Evaluation

[ ] Deployment (API or Web App)

⚖️ License
Initially was a kaggle dataset, this version was "cleaned" for assignment through UNISA

🏆 Project Milestone: Multi-Model Evaluation
In this phase, I developed and compared three distinct architectures to classify toxic comments using a dataset of 150,000+ entries.

Baseline (Logistic Regression + TF-IDF): Established a strong starting point with a 0.76 F1-Score, proving that simple word-frequency models are highly effective for toxicity detection.

Standard Neural Network (Champion): By implementing a Deep Learning model with an Embedding layer and tuning the decision threshold to 0.2, I achieved the best balance with an F1-Score of 0.77 and a significantly higher Precision (0.83).

Word2Vec + NN: Experimented with pre-trained semantic embeddings. While this model achieved the highest Recall (0.81), it proved overly sensitive for this specific use case, resulting in more false positives.

Current Status: The Standard Neural Network is saved as the production-ready model for deployment.


---

