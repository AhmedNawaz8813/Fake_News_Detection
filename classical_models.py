import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay

import joblib
import os


# ==========================================
# 1. Load Data and Train-Test Split
# ==========================================
print("Loading sentiment_dataset.csv...")
df = pd.read_csv('data/cleaned/sentiment_dataset.csv')

# Handle any unexpected empty text strings safely
df['cleaned_text'] = df['cleaned_text'].fillna('')

# Split into training and testing sets (80% train, 20% test)
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df[['cleaned_text', 'sentiment_score']], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

# ==========================================
# 2. Feature Extraction: TF-IDF + Sentiment
# ==========================================
print("Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_train_tfidf = vectorizer.fit_transform(X_train_text['cleaned_text'])
X_test_tfidf = vectorizer.transform(X_test_text['cleaned_text'])

train_sentiment = X_train_text['sentiment_score'].values.reshape(-1, 1)
test_sentiment = X_test_text['sentiment_score'].values.reshape(-1, 1)

# Combined Sparse Matrices (Shifting VADER from [-1,1] to [0,2] for Naive Bayes)
X_train_combined = sp.hstack((X_train_tfidf, train_sentiment + 1.0))
X_test_combined = sp.hstack((X_test_tfidf, test_sentiment + 1.0))

# ==========================================
# 3. Model Training and Evaluation
# ==========================================
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
}

classical_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_combined, y_train)
    predictions = model.predict(X_test_combined)

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='binary')
    proba = model.predict_proba(X_test_combined)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    classical_results[name] = {"Accuracy": acc, "F1-Score": f1, "ROC-AUC": roc_auc}

    print(f"--- Evaluation for {name} ---")
    print(classification_report(y_test, predictions, target_names=['Fake (0)', 'Real (1)']))

    # ✅ Now inside the loop — runs for every model
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions,
        display_labels=['Fake', 'Real'],
        ax=ax
    )
    ax.set_title(f'Confusion Matrix: {name}')
    plt.tight_layout()
    plt.show()

# Convert results dictionary to a structured DataFrame
summary_df = pd.DataFrame(classical_results).T.reset_index()
summary_df.rename(columns={'index': 'Model'}, inplace=True)

print("\n=== Classical Model Performance Summary ===")
print(summary_df)

# ==========================================
# 4. Metric Visualization (Matplotlib & Seaborn)
# ==========================================
print("\nGenerating performance visualization...")

# Melt the DataFrame to make it compatible with seaborn's grouping
melted_df = pd.melt(summary_df, id_vars="Model", var_name="Metric", value_name="Score")

# Set up the plotting window
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Create a grouped bar chart
ax = sns.barplot(x="Model", y="Score", hue="Metric", data=melted_df, palette="muted")

# Add chart labels and title
plt.title("Comparison of Baseline Classical Models", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Machine Learning Model", fontsize=12, labelpad=10)
plt.ylabel("Performance Score (0.0 - 1.0)", fontsize=12, labelpad=10)
plt.ylim(0, 1.1)  # Set limit slightly above 1.0 to leave room for value labels

plt.legend(title='Metric', loc='lower right', fontsize=10, title_fontsize=11)

# Display exact scores on top of each bar
for p in ax.patches:
    height = p.get_height()
    if height > 0: # Ensure we don't label empty bars
        ax.annotate(f'{height:.4f}',
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='center',
            xytext=(0, 8),
            textcoords='offset points',
            fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions,
    display_labels=['Fake', 'Real'],
    ax=ax
)
ax.set_title(f'Confusion Matrix: {name}')
plt.tight_layout()
plt.show()




os.makedirs('models', exist_ok=True)
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
for name, model in models.items():
    safe_name = name.replace(' ', '_').lower()
    joblib.dump(model, f'models/{safe_name}.pkl')