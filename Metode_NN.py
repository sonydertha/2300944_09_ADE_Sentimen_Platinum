# import library
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# from Sastrawi import POSTagger
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import string
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from cleansing import text_cleansing


# import raw dataset for preprocessing
df = pd.read_csv('train_preprocess.tsv.txt',
                 sep ='\t',
                 names = ["text", "label"])


# teks cleansing dilakukan terpisah karena makan waktu untuk ganti kata alay ke kata formal
# df['clean_text'] = df.text.apply (text_cleansing)
# df.head()

# import dataset with clean_text

# Create a function to tokenize the text
def word_tokenize(text):
  words = nltk.word_tokenize(text)
  return words

# # Create a function to tokenize the text using word_tokenize
def tokenize_sent(text):
  sent = nltk.sent_tokenize(text)
  return sent

# # get Indonesian stopword
stop_words = nltk.corpus.stopwords.words("indonesian")

# # function to remove stop words
def filter_stopwords (tokens) :
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

# # Apply the tokenization function to `clean_text` column
# df["tokens"] = df["clean_text"].apply(tokenize)
# df["tokens_sent"] = df["clean_text"].apply(tokenize_sent)
# # Remove the stop words from the tokens
# df["filtered_tokens"] = df["tokens"].apply(filter_stopwords)

# Initialize Sastrawi Stemmer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Define a function preproses tokens to be a preprossesed text
def stemming(tokens):
 stemmed_tokens = [stemmer.stem(token) for token in tokens]
 return ' '.join(stemmed_tokens)   

# apply stemmer dimatikan karena sangat lama
# df['preprocessed_text'] = df['filtered_tokens'].apply(stemming)

    ## FEATURE EXTRACTION 

# import preprocessed dataset 
df_ready = pd.read_csv("sentiment_analysis_dataset_filtered_tokens.csv")
df_ready['preprocessed_text'] = df_ready['preprocessed_text'].fillna('')

# feature extraction
# perpare preprocessed and tokenized text data
tokenized_texts = df_ready['preprocessed_text']


# perpare preprocessed and tokenized text data
tokenized_texts = df_ready['preprocessed_text']

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the data to obtain TF-IDF features from text
tfidf_features = tfidf_vectorizer.fit_transform(tokenized_texts)


# Split train and test
X_train, X_test, y_train, y_test = train_test_split(
   tfidf_features, df_ready['label']
   , test_size=0.2, random_state=42
   )

# tackle imbalance 
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    ## Training with MLP

 # initialize  MLP Classifier
mlp_classifier = MLPClassifier()

# Fit the model on the training data
mlp_classifier.fit(X_train_smote, y_train_smote)

# Predict on the test data
predictions_mlp = mlp_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy_logreg = accuracy_score(y_test, predictions_mlp)
precision_logreg = precision_score(y_test, predictions_mlp, average='weighted')
recall_logreg = recall_score(y_test, predictions_mlp, average='weighted')
f1_logreg = f1_score(y_test, predictions_mlp, average='weighted')


# Save the tuned logistic regression model 
model_mlp_filename = 'tuned_mlp_model.jl'
with open(model_mlp_filename, 'wb') as model_file:
    joblib.dump(mlp_classifier, model_mlp_filename)

# Load the tuned logistic regression model 
with open(model_mlp_filename, 'rb') as model_file:
    loaded_model_mlp = joblib.load(model_mlp_filename)


## save and load tfidf_vectorizer
# Save the tfidf_vectorizer used for training
tfidf_vectorizer_filename = 'tfidf_vectorizer.pkl'
with open(tfidf_vectorizer_filename, 'wb') as vectorizer_file:
   joblib.dump(tfidf_vectorizer, vectorizer_file)

# Load the tfidf_vectorizer used for training
with open(tfidf_vectorizer_filename, 'rb') as vectorizer_file:
   loaded_tfidf_vectorizer = joblib.load(tfidf_vectorizer_filename)

# Fit the loaded model on the training data
loaded_model_mlp.fit(X_train_smote, y_train_smote)

def text_processing(text):
   processed_text = text_cleansing(text)
   processed_text = nltk.word_tokenize(processed_text)
   processed_text = filter_stopwords(processed_text)
   processed_text = stemming(processed_text)
   return processed_text

def predict_NN (text):
   tfidf_vector = loaded_tfidf_vectorizer.transform([text])  
   sentiment = loaded_model_mlp.predict(tfidf_vector)
   return sentiment 

def predict_NN_files (file_upload):
   # read csv file upload, jika eror dengan metode biasa
    df_NN_upload = pd.DataFrame(file_upload.iloc[:,[0]])
    # rename kolom menjadi "raw_text"
    df_NN_upload.columns = ["raw_text"]
    # processing texts on file
    df_NN_upload["processed_text"] = df_NN_upload["raw_text"].apply(text_processing)
    print ("Text Processing done!")
    # predict Sentiment 
    df_NN_upload["Sentiment"] = df_NN_upload["processed_text"].apply(predict_NN)
    print ("Sentiment Analysis Done")
    return df_NN_upload

# Print the predicted label

# print(f"sentimennya adalah : {predict_NN(new_text_processed)}")