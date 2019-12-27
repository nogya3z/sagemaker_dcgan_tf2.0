import argparse
import boto3
import numpy as np
import os
import requests
import tensorflow as tf
import zipfile

from PIL import Image


class download_unzip_upload:
    
    s3 = boto3.client('s3')
    
    def __init__(self, file_id, destination, output_file, bucket_name):
        self.file_id = file_id
        self.destination = destination
        self.output_file = output_file
        self.bucket_name = bucket_name
        
        
    def download_file_from_google_drive(self):
        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(self):
            CHUNK_SIZE = 32768
            a  = self.destination
            with open(a, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)

        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params = { 'id' : self.file_id }, stream = True)
        token = get_confirm_token(response)

        if token:
            params = { 'id' : self.file_id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        save_response_content(self)
        
          
    def unzip(self):
        with zipfile.ZipFile(self.destination, 'r') as zip_ref:
                zip_ref.extractall(self.destination[:-4])

    def file_directories(self, folder):
        train_list = [os.path.join(folder, i) for i in os.listdir(folder)]
        return(train_list)                
                

    def _bytes_feature(self, value):
      """Returns a bytes_list from a string / byte."""
      if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def image_example(self, image_string):
      feature = {
          'image': self._bytes_feature(image_string),
      }
      return tf.train.Example(features=tf.train.Features(feature=feature))
    
    def convert_to_tfrecord(self, input_files):
      """Converts a file to TFRecords. We also preprocess the image and reduce file size"""
      print('Generating %s' % self.output_file)
      with tf.io.TFRecordWriter(self.output_file) as record_writer:
          for file in input_files:
            img = Image.open(file)
            img = img.resize((64,64))
            img = np.asarray(img)
            img  = tf.io.encode_jpeg(img)
            example = self.image_example(img)
            record_writer.write(example.SerializeToString())
    
    
    def run_all(self):
      self.download_file_from_google_drive()
      self.unzip()
      file_location = os.path.join(self.destination[:-4], 'img_align_celeba')
      input_files = self.file_directories(file_location)
      self.convert_to_tfrecord(input_files)
      self.s3.upload_file(self.output_file, self.bucket_name, "work_folder" + "/" + self.output_file)
    

