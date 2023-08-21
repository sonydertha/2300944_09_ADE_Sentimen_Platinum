import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# from Sastrawi import POSTagger
import re
import joblib
import numpy as np
from cleansing import cleansing

# Create a function to tokenize the text
def word_tokenize(text):
  words = nltk.word_tokenize(text)
  return words

# # get Indonesian stopword
stop_words = nltk.corpus.stopwords.words("indonesian")

# function to remove stop words
def filter_stopwords (tokens) :
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

# Initialize Sastrawi Stemmer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
# function preproses tokens to be a preprossesed text
def stemming(tokens):
 stemmed_tokens = [stemmer.stem(token) for token in tokens]
 return ' '.join(stemmed_tokens) 


# Load the tuned logistic regression model 
model_mlp_filename = 'mlp_model.pkl'
with open(model_mlp_filename, 'rb') as model_file:
    loaded_model_mlp = joblib.load(model_mlp_filename)

# Load the tfidf_vectorizer used for training
tfidf_vectorizer_filename = 'tfidf_vectorizer.pkl'
with open(tfidf_vectorizer_filename, 'rb') as vectorizer_file:
   loaded_tfidf_vectorizer = joblib.load(tfidf_vectorizer_filename)

def text_processing_NN(text):
   processed_text = cleansing(text)
   processed_text = nltk.word_tokenize(processed_text)
   processed_text = filter_stopwords(processed_text)
   processed_text = stemming(processed_text)
   return processed_text

sentiment = ['negative', 'neutral', 'positive']

def predict_NN (text):
   tfidf_vector = loaded_tfidf_vectorizer.transform([text])  
   prediction = loaded_model_mlp.predict(tfidf_vector)
   get_sentiment = sentiment[np.argmax(prediction[0])]
   return get_sentiment 

def predict_NN_files (file_upload):
   # read csv file upload, jika eror dengan metode biasa
    df_NN_upload = pd.DataFrame(file_upload.iloc[:,[0]])
    # rename kolom menjadi "raw_text"
    df_NN_upload.columns = ["raw_text"]
    # processing texts on file
    df_NN_upload["clean_text"] = df_NN_upload["raw_text"].apply(text_processing_NN)
    print ("Text Processing done!")
    # predict Sentiment 
    df_NN_upload["Sentiment"] = df_NN_upload["clean_text"].apply(predict_NN)
    print ("Sentiment Analysis Done")
    return df_NN_upload
