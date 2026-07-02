import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import os


# 1. Download required NLTK data (only needs to be run once)
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



# 2. Define the Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# ==========================================
# 3. Execution Pipeline
# ==========================================

print("Loading raw CSV files...")
# Load Fake.csv (and True.csv to create a complete binary dataset)
df_fake = pd.read_csv('data/raw/Fake.csv')
df_true = pd.read_csv('data/raw/True.csv') # Assuming you have this downloaded as well

# Add the target labels (0 for Fake, 1 for True)
df_fake['label'] = 0
df_true['label'] = 1

# Combine them into one master dataframe
df_master = pd.concat([df_fake, df_true], ignore_index=True)
df_master['full_text'] = df_master['title'] + " " + df_master['text']  
df_master = df_master.drop_duplicates(subset=['full_text'])            
df_master = df_master.sample(frac=1, random_state=42).reset_index(drop=True)

print("Cleaning text data. This might take a few minutes...")
# Apply the cleaning function (this is CPU intensive and takes time)
df_master['cleaned_text'] = df_master['full_text'].apply(clean_text)

# Drop any rows where the text became completely empty after cleaning
df_master = df_master[df_master['cleaned_text'].str.strip() != '']

os.makedirs('data/cleaned', exist_ok=True)

# Keep only the columns necessary for the next step to save memory
df_final = df_master[['full_text', 'cleaned_text', 'label']]

print("Saving cleaned data...")
df_final.to_csv('data/cleaned/cleaned_dataset.csv', index=False)
print("Done! Data is ready for Step 02.")