
# Explore Data Notebook #

#First import necessary libraries as virtual env 
import pandas as pd

# Load the dataset
# Make sure your file is in a folder named 'data'
try:
    df = pd.read_csv('data/Toxic.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Toxic.csv not found in the 'data' folder.")

# Look at the first 5 rows
print("\n--- First 5 Rows ---")
print(df.head())

# See the categories of toxicity
print("\n--- Column Names (Toxicity Categories) ---")
print(df.columns)

# Check how many comments are actually toxic
print("\n--- Summary Statistics ---")
print(df.describe())