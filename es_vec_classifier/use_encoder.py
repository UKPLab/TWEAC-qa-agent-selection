import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
import requests

class USEEncoder:
    def __init__(self, url):
        self.embedder = hub.load(url)

    def encode(self, sentences, **kwargs):
        return self.embedder.signatures['question_encoder'](input=tf.constant(sentences))["outputs"]

class USEEncoderAPI:
    def encode(self, sentences, **kwargs):
        resp = requests.post('http://localhost:5003/use', json=sentences)
        resp = resp.json()
        return np.array(resp['vector'])