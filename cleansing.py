""" 
Function untuk membersihkan data text
"""
import re
import pandas as pd
from db import get_abusive_data, create_connection

def text_cleansing(text):
    # Bersihkan tanda baca (selain huruf dan angka)
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # ubah teks menjadi lower case
    clean_text = clean_text.lower()
    # menghilangkan emoji
    clean_text = re.sub(r'xf0\S+', '', clean_text)
    clean_text = re.sub(r'xe\S+', '', clean_text)
    #remove repeated character
    clean_text = re.sub(r'(.)\\1+', r'\1', clean_text)

    # # abusive  # tidak menggunakkan kamus abusive untuk analisis sentimen
    # conn = create_connection()
    # df_abusive = get_abusive_data(conn)
    # abusive_words = df_abusive['word'].tolist()
    # for word in abusive_words:
    #     clean_text = clean_text.replace(word, '***')

    # menggantikan kata alay dengan kata formal
    replacement_words = pd.read_csv('csv_data/new_kamusalay.csv')
    replacement_dict = dict(zip(replacement_words['alay_word'], replacement_words['formal_word']))
    words = clean_text.split()
    replaced_words = [replacement_dict.get(word, word) for word in words]
    clean_text = ' '.join(replaced_words)
    # menghilangkan spasi di awal dan akhir teks
    clean_text = clean_text.strip()
    return clean_text

def cleansing_files(file_upload):
    # read csv file upload, jika eror dengan metode biasa
    df_upload = pd.DataFrame(file_upload.iloc[:,[0]])
    # rename kolom menjadi "raw_text"
    df_upload.columns = ["raw_text"]
    # bersihkan teks menggunakan fungsi cleansing
    # simpan di kolom "clean_text"
    df_upload["clean_text"] = df_upload["raw_text"].apply(text_cleansing)
    # mensensor kata abusive sesuai dari data dalam bentuk csv
    
    print("Cleansing text succes!")
    return df_upload