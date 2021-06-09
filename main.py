import requests
import tensorflow as tf 
from google.cloud import storage
import numpy as np 
import pandas as pd 
import os
import glob
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def transform(age_input, bmi_input, children_input, sex_input, smoker_input, region_input):
  #declare intrinsic value
  age_max = 64 
  age_min = 18
  bmi_max = 53.13
  bmi_min = 15.96
  children_max = 5
  children_min = 0
  
  #DECLARE VARIABEL
  #age
  age	= float((age_input-age_min) / (age_max-age_min))
  #bmi
  bmi	= float((bmi_input-bmi_min)/(bmi_max-bmi_min))
  #children
  children = float((children_input-children_min)/(children_max-children_min))
  #sex
  sex_female	= 0.0
  sex_male	= 0.0
  if (sex_input=='female'):
    sex_female = float(1)
  else: sex_male = float(1)
  #smoker
  smoker_no	= 0.0
  smoker_yes = 0.0
  if (smoker_input == "no"):
    smoker_no = float(1)
  else: smoker_yes = float(1)
  #region
  region_northeast = 0.0
  region_northwest = 0.0
  region_southeast = 0.0	
  region_southwest = 0.0
  if region_input == "northeast":
    region_northeast = float(1)
  elif region_input == "northwest":
    region_northwest = float(1)
  elif region_input == "southeast":
    region_southeast = float(1)
  else :
    region_southwest = float(1)

  return np.array([[age, bmi, children, sex_female,sex_male,smoker_no, smoker_yes,
                  region_northeast,region_northwest,region_southeast,region_southwest]])
  
def predict(requests):
    request_json = requests.get_json()
    age = float(request_json['age'])
    bmi = float( request_json['bmi'])
    children = int(request_json['children'])
    sex = request_json['sex']
    smoker = request_json['smoker']
    region = request_json['region']

    hasil_transform = transform(age,bmi,children,sex,smoker,region)

    ## LOAD MODEL DARI BUCKET
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('bangkit-cap0134-model')
    # ## MEMBUAT DIREKTORI TEMP/MYMODEL UNTUK PENGGUNAAN SEMENTARA
    # file_path = '/tmp/my_model'
    # # pastikan directory ada
    # os.makedirs(os.path.dirname(file_path), exist_ok=True)

    model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
    ])

    blob_weight1 = bucket.blob('my_model/variables/variables.data-00000-of-00001')
    blob_weight2 = bucket.blob('my_model/variables/variables.index')

    blob_weight1.download_to_filename('/tmp/variables.data-00000-of-00001')
    blob_weight2.download_to_filename('/tmp/variables.index')

    model_3.load_weights('/tmp/variables')

    hasil_prediksi = model_3.predict(hasil_transform)
    hasil_prediksi_string = str(hasil_prediksi[0][0])


    return(hasil_prediksi_string)