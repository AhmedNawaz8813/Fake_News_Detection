# Comprehensive Breakdown of the Fake News Detection Pipeline

This document provides an elaborate explanation of the entire Machine Learning pipeline used to detect fake news detection. The workflow is divided into 5 distinct stages: 
* Data Preparation and cleaning
* Sentiment Engineering
* Classical Machine Learning
    * Naive Bayes
    * Logistic Regression
    * Random Forest
* Training a Deep Learning Neural Network
* Transformer fine tuning


## Phase 1: Data Preperation and Cleaning

Libraries Used
* `pandas`: For data manipulation.
* `re` for Regular Expressions
* `nltk` for natural language processing tasks

Resources Downloaded
* `stopwords`
* `wordnet` 

The script loads two raw datasets `Fake.csv` and `True.csv`. Each downloaded from Kaggle: [ISOT Fake and True News Dataset](https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset)
In the Dataset; 
* 0 is assigned to fake news
* 1 is assigned to true news


The function `clean_text` is defined and implemented to do the following:
* Convert text to lowercase.
* Remove URL's
* Strip out HTML Tags
* Delete non alphabetical characters
* Split text into words
* Filters out English stop words
* Applies WordNet Lemmatization to reduce words to their base form.

The two dataframes are concatenated into a single master dataframe, and a new `full_text` column is created by combining the articles title and the body text followed by it.
Duplicate articles are removed based on the `full_text` column, after which the entire dataset is shuffled, to introduce randomness.
A new `cleaned_text` column is created. Any rows which are empty, as a result of the cleaning process are dropped.

The final dataset `cleaned_dataset.csv` will now have the following columns:
* full_text
* cleaned_text
* label


## Phase 2: Sentiment Feature Engineering and EDA
This phase applies sentiment polarity scores to the dataset and visually explores the data

*   **VADER Initialization**: The code imports and downloads the `vader_lexicon` from NLTK to perform rule-based sentiment analysis
*   **Sentiment Scoring**: A custom function applies the `SentimentIntensityAnalyzer` to the uncleaned `full_text` of each article to extract the `compound` sentiment score. This score ranges from -1.0 (extremely negative) to +1.0 (extremely positive).
*   **Data Export**: The dataset, now containing the new `sentiment_score` feature, is saved as `sentiment_dataset.csv` for use in downstream modeling.
*   **Exploratory Data Analysis (EDA)**: The script utilizes `matplotlib` and `seaborn` to generate visual comparisons of sentiment between fake and real news.
*   **Visualizations**: It generates a two panel plot containing a histogram, which shows the distribution curve of sentiment scores, and a boxplot, which highlights the statistical spread and outliers for both classes.
*   **Statistical Output**: The code calculates and prints the mathematical average of the sentiment scores grouped by their authenticity labels, revealing a slight negative average for fake news and a positive average for real news.

---

## Phase 3: Classical Machine Learning Pipeline
This phase establishes a performance baseline using traditional machine learning algorithms and TF-IDF vectorization

*   **Data Loading**: The script reads `sentiment_dataset.csv` and proactively fills any missing `cleaned_text` entries with empty strings to prevent processing errors.
*   **Train-Test Split**: The data is split into an 80% training set and a 20% testing set, utilizing stratified sampling to maintain an exact balance of fake and real labels in both sets.
*   **TF-IDF Vectorization**: A `TfidfVectorizer` transforms the cleaned text into numerical features, capped at the top 5,000 most frequent terms and utilizing both unigrams and bigrams.
*   **Feature Fusion**: The VADER sentiment scores are mathematically shifted from a [-1.0, 1.0] range to a strictly positive [0.0, 2.0] range, to avoid negative values for Naive Bayes. These scores are horizontally stacked with the TF-IDF sparse matrix.
*   **Model Training**: Three independent models are initialized and trained: Multinomial Naive Bayes, Logistic Regression (configured with a maximum of 1000 iterations), and a Random Forest Classifier (configured with 100 estimators and a max depth of 20).
*   **Evaluation Metrics**: Each model predicts outcomes on the test set and calculates Accuracy, binary F1-Score, and ROC-AUC metrics.
*   **Confusion Matrices**: The script dynamically generates and displays a visual Confusion Matrix for each trained model using `ConfusionMatrixDisplay`.
*   **Performance Visualization**: The evaluation metrics are structured into a pandas DataFrame, "melted" for compatibility, and plotted as a grouped Seaborn bar chart to provide a clear comparison of baseline performances.
*   **Model Serialization**: The trained vectorizer and all three classical machine learning models are saved as `.pkl` files in a `models/` directory using `joblib` for future deployment.

---

## Phase 4: Deep Learning (MLP with GloVe Embeddings)
This phase introduces deep learning by constructing a Multilayer Perceptron (MLP) neural network leveraging pre-trained word vectors.

*   **TensorFlow Setup**: The script imports Keras components and suppresses background TensorFlow logging warnings for cleaner output.
*   **Data Preparation**: It loads the dataset and executes an 80/20 stratified split.
*   **Dynamic Sequence Sizing**: The code calculates the length of every article and identifies the 95th percentile. This mathematically determines an optimal sequence length of 531 words, ensuring 95% of articles are preserved without arbitrary truncation.
*   **Tokenization**: A Keras `Tokenizer` is fitted to learn the top 20,000 words, converting the text into sequences of integer IDs. The sequences are then padded or truncated to exactly 531 tokens.
*   **GloVe Integration**: Pre-trained 100-dimensional GloVe word embeddings are loaded from a local text file. An embedding matrix is specifically constructed to map the dataset's vocabulary to these Stanford-trained geometric vectors.
*   **Network Architecture**: A `Sequential` model is built, starting with an `Embedding` layer that uses the GloVe matrix as frozen, non-trainable weights. 
*   **Pooling and Dense Layers**: A `GlobalAveragePooling1D` layer compresses the sequence of word vectors into a single summary vector. This feeds into dense decision-making layers with `Dropout` applied at 50% to prevent the network from memorizing the training data.
*   **Output and Compilation**: The final layer uses a sigmoid activation to output a probability between 0.0 and 1.0, and the model is compiled using the Adam optimizer and binary crossentropy loss.
*   **Training and Evaluation**: The model is trained over 5 epochs utilizing a 10% validation split to monitor overfitting. Finally, predictions are generated on the test set, producing an Accuracy score, an F1-Score, and a detailed classification report.

---

## Phase 5: State-of-the-Art Transformer Fine-Tuning
The final phase utilizes Hugging Face tools to fine-tune a pre-trained BERT model directly in a Google Colab environment.

*   **Environment Configuration**: The notebook installs the necessary `transformers`, `datasets`, and `accelerate` libraries required for Hugging Face integration.
*   **Data Formatting**: After loading the CSV and dropping rows with missing labels, the pandas DataFrames are converted into Hugging Face `Dataset` objects and grouped into a `DatasetDict`.
*   **BERT Tokenization**: The script downloads the official `bert-base-uncased` tokenizer. It defines a function to apply padding and strict truncation to enforce BERT's hard limit of 512 tokens. This function is mapped across the entire dataset concurrently.
*   **Custom Evaluation Metric**: A `compute_metrics` function is defined to process the model's raw logits using `argmax`, calculating both accuracy and F1-score during the evaluation phase.
*   **Model Initialization**: The `AutoModelForSequenceClassification` class downloads the 110-million parameter BERT model, specifically configuring it for a binary classification task by setting `num_labels=2`.
*   **Training Configuration**: `TrainingArguments` are established to train the model over 3 epochs with an evaluation strategy executed at the end of each epoch. Batch sizes are restricted to 8 to prevent Out-Of-Memory (OOM) errors, and a tiny learning rate of 2e-5 is applied alongside weight decay.
*   **Execution**: The Hugging Face `Trainer` API takes control of the training loop, fine-tuning the model and running a final evaluation to log the performance metrics.

