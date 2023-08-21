"""

Flask API Application

"""
from flask import Flask, jsonify, request
from flasgger import Swagger, swag_from, LazyJSONEncoder, LazyString
import pandas as pd
import numpy as np
from time import perf_counter
import re
import flask
from cleansing import cleansing
from db import (
    create_connection, 
    insert_result_to_db, show_analisis_result,
    insert_upload_result_to_db
)
from LSTM import lstm, analisis_file
from NeuralNetwork import text_processing_NN, predict_NN, predict_NN_files

# initializze flask application
app = Flask(__name__)

#Assign LazyJSONEncoder to app.json_encoder for swagger UI
app.json_encoder = LazyJSONEncoder
# create swagger config & swagger template
Swagger_template ={
    "info":{
        "title": LazyString(lambda: "Analisis sentimen dalam teks menggunakan LSTM dan NeuralNetwork"),
        "version": LazyString(lambda: "1.0.0"),
        "description": LazyString(lambda: "Dokumentasi API untuk Analisis Sentimen Menggunakan LSTM dan NeuralNetwork")
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
        "author": "Adelia Christyanti dan Sony Dertha Setiawan"
    }
    return jsonify(welcome_msg)

# Show analisis result
@swag_from('docs/show_analisis_result.yml', methods=['GET'])
@app.route('/show_analisis_result', methods=['GET'])
def show_analisis_result_api():
    db_connection = create_connection()
    analisis_result = show_analisis_result(db_connection)
    return jsonify(analisis_result)

#cleansing text using form
@swag_from('docs/lstm.yml', methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm_endpoint():
    # get text from input user
    raw_text = request.form["raw_text"]
    # cleansing text
    start = perf_counter()
    clean_text = cleansing(raw_text)
    sentiment = lstm(clean_text)
    end = perf_counter()
    time = end - start
    model = "LSTM"
    print(f'processing time :{time}')
    result_response ={"raw_text": raw_text,
                       "clean_text": clean_text,
                       "Sentiment": sentiment, 
                       "Processing time": time, 
                       "Model": model}
    # insert result to database
    db_connection = create_connection()
    insert_result_to_db(db_connection, raw_text, clean_text, sentiment, model)
    return jsonify(result_response)

# Cleansing text using csv upload
@swag_from('docs/lstm_upload.yml', methods=['POST'])
@app.route('/lstm_upload', methods=['POST'])
def cleansing_upload():
    # Get file from upload to dataframe
    uploaded_file = request.files['upload_file']
    # Read csv file to dataframe
    df_upload = pd.read_csv(uploaded_file,encoding ='latin-1').head(1000)
    print('Read dataframe Upload success!')
    start = perf_counter()
    df_cleansing = analisis_file(df_upload)
    end = perf_counter()
    time = end - start
    model = "LSTM"
    print(f'processing time :{time}')
    
    # Upload result to database
    db_connection = create_connection()
    insert_upload_result_to_db(db_connection, df_cleansing, model)    
    print("Upload result to database success!")
    result_response = df_cleansing.to_dict(orient='records')
    return jsonify(result_response)

# sentimen analysis with NeuralNetwork using form
@swag_from("docs/NeuralNetwork.yml", methods=['POST']) 
@app.route('/NeuralNetwork', methods=['POST'])
def NeuralNetwork():
    raw_text = request.form.get('text')
    clean_text = text_processing_NN(raw_text)
    sentiment = predict_NN(clean_text)
    model = "NeuralNetwork"
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using LSTM",
        'data':{
            'raw text': raw_text,
            'clean text':clean_text,
            'sentiment': sentiment,
            'Model': model
        },
    }
    response_data = jsonify(json_response)
    # insert result to database
    db_connection = create_connection()
    insert_result_to_db(db_connection, raw_text, clean_text, sentiment, model)
    return response_data

# sentimen analysis with NeuralNetwork using csv upload
@swag_from('docs/NeuralNetwork_upload.yml', methods=['POST'])
@app.route('/NeuralNetwork_upload', methods=['POST'])
def NeuralNetwork_upload():
    # Get file from upload to dataframe
    uploaded_file = request.files['upload_file']
    # Read csv file to dataframe
    df_upload = pd.read_csv(uploaded_file,encoding ='latin-1')
    print('Read dataframe Upload success!')
    start = perf_counter()
    df_predict_NN = predict_NN_files(df_upload)
    end = perf_counter()
    time = end - start
    model = "NeuralNetwork"
    print(f'processing time :{time}')
    
    # Upload result to database 
    # define new connection
    db_connection = create_connection()
    insert_upload_result_to_db(db_connection, df_predict_NN, model)    
    print("Upload result to database success!")
    result_response = df_predict_NN.to_dict(orient='records')
    return jsonify(result_response)


if __name__ == '__main__':
    app.run()