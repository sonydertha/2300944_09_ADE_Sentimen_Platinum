"""
Function untuk membersihkan data teks
"""
import re
import pandas as pd
import pickle, re
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def cleansing(text):
    # Bersihkan tanda baca (selain huruf dan angka)
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # ubah teks menjadi lower case
    clean_text = clean_text.lower()
    # menghilangkan emoji
    clean_text = re.sub(r'xf0\S+', '', clean_text)
    clean_text = re.sub(r'xe\S+', '', clean_text)
    #remove repeated character
    clean_text = re.sub(r'(.)\1+', r'\1', clean_text)
    # menggantikan kata alay dengan kata formal
    replacement_words = pd.read_csv('csv_data/new_kamusalay.csv')
    replacement_dict = dict(zip(replacement_words['alay_word'], replacement_words['formal_word']))
    words = clean_text.split()
    replaced_words = [replacement_dict.get(word, word) for word in words]
    clean_text = ' '.join(replaced_words)
    # menghilangkan spasi di awal dan akhir teks
    clean_text = clean_text.strip()
    return clean_text


