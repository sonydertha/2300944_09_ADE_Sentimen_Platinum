"""
Flask API Application
"""

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from time import perf_counter
from flasgger import Swagger, swag_from, LazyString, LazyJSONEncoder
from cleansing import text_cleansing, cleansing_files
from db import (
    create_connection, insert_dictionary_to_db, 
    insert_result_to_db, show_cleansing_result,
    insert_upload_result_to_db
)
import flask
flask.json.provider.DefaultJSONProvider.sort_keys = False
import pickle, re
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Metode_NN import  text_processing, predict_NN

# Set Up Database
db_connection = create_connection()
insert_dictionary_to_db(db_connection)
db_connection.close()
# initializze flask application
app = Flask(__name__)

#Assign LazyJSONEncoder to app.json_encoder for swagger UI
app.json_encoder = LazyJSONEncoder
# create swagger config & swagger template
Swagger_template ={
    "info":{
        "title": LazyString(lambda: "API untuk analisis sentimen"),
        "version": LazyString(lambda: "1.0.0"),
        "description": LazyString(lambda: "Dokumentasi API untuk analisis sentimen")
    },
    "host": LazyString(lambda: request.host)
}
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
# intiliazeSwagger from tempalte & config
swagger = Swagger(app, template=Swagger_template, config=swagger_config)

# homepage
@swag_from('docs/home.yml', methods=['GET'])
@app.route('/', methods=['GET'])
def home():
    welcome_msg = {
        "version": "1.0.0",
        "message": "Welcome to Flask API",
        "author": "Adelia Christyanti dan Sony Dermawan"
    }
    return jsonify(welcome_msg)

# Show cleansing result
@swag_from('docs/show_cleansing_result.yml', methods=['GET'])
@app.route('/show_cleansing_result', methods=['GET'])
def show_cleansing_result_api():
    db_connection = create_connection()
    cleansing_result = show_cleansing_result(db_connection)
    return jsonify(cleansing_result)

#cleansing text using form
@swag_from('docs/cleansing_form.yml', methods=['POST'])
@app.route('/cleansing_form', methods=['POST'])
def cleansing_form():
    # get text from input user
    raw_text = request.form["raw_text"]
    # cleansing text
    start = perf_counter()
    clean_text = text_cleansing(raw_text)
    end = perf_counter()
    time = end - start
    print(f'processing time :{time}')
    result_response ={"raw_text": raw_text, "clean_text": clean_text, "processing time": time}
    # insert result to database
    db_connection = create_connection()
    insert_result_to_db(db_connection, raw_text, clean_text)
    return jsonify(result_response)

# Cleansing text using csv upload
@swag_from('docs/cleansing_upload.yml', methods=['POST'])
@app.route('/cleansing_upload', methods=['POST'])
def cleansing_upload():
    # Get file from upload to dataframe
    uploaded_file = request.files['upload_file']
    # Read csv file to dataframe
    df_upload = pd.read_csv(uploaded_file,encoding ='latin-1').head(1000)
    print('Read dataframe Upload success!')
    start = perf_counter()
    df_cleansing = cleansing_files(df_upload)
    end = perf_counter()
    time = end - start
    print(f'processing time :{time}')
    
    # Upload result to database
    db_connection = create_connection()
    insert_upload_result_to_db(db_connection, df_cleansing)    
    print("Upload result to database success!")
    result_response = df_cleansing.to_dict(orient='records')
    return jsonify(result_response)

max_features = 96
# tokenizer = Tokenizer(num_words=max_features, split =' ', lower=True) # ini nanti diganti dari jupyter lab
file = open('tokenizer/tokenizer.pickle','rb') # tokenizer
tokenizer = pickle.load(file)
file.close()
sentiment = ['negative', 'neutral', 'positive']

# file = open('resources_of_lstm/x_pad_sequences.pickle','rb')
# pad_sequences = pickle.load(file)
# file.close()

def cleansing(sent):
    string = sent.lower()
    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
    string = string.strip()
    return string

#file_token = load_model('tokenizer/tokenizer.pickle')

model_file_from_lstm = load_model('model_of_lstm/model.h5')

@swag_from("docs/lstm.yml", methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():
    original_text = request.form.get('text')
    text = [cleansing(original_text)]
    # X = tokenizer.texts_to_sequences(total_data)
    feature = tokenizer.texts_to_sequences(text)# pakai dari jupyter lab
    feature = pad_sequences(feature, maxlen=max_features)


    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using LSTM",
        'data':{
            'raw text': original_text,
            'clean text':text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

# sentimen analysis with NeuralNetwork
@swag_from("docs/NeuralNetwork.yml", methods=['POST']) ## bikin yml file
@app.route('/NeuralNetwork', methods=['POST'])
def NeuralNetwork():
    original_text = request.form.get('text')
    text = text_processing(original_text)
    result = predict_NN(text)
    sentiment_list = result.tolist()
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using LSTM",
        'data':{
            'raw text': original_text,
            'clean text':text,
            'sentiment': sentiment_list
        },
    }
    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run(debug=True)