from __future__ import absolute_import
from django.conf import settings

import sys
import argparse
import logging

import tensorflow as tf

from .model.model import Model

from .util import dataset
from .util.data_gen import DataGen
from .util.export import Exporter

tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename='aocr.log')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def load_model(sess):
    model = Model(
            phase='predict',
            visualize=False,
            output_dir='results',
            batch_size=1,
            initial_learning_rate=1.0,
            steps_per_checkpoint=0,
            model_dir='\modelss\\',
            target_embedding_size=10,
            attn_num_hidden=128,
            attn_num_layers=2,
            clip_gradients=True,
            max_gradient_norm=5.0,
            session=sess,
            load_model=True,
            gpu_id=0,
            use_gru=False,
            use_distance=True,
            max_image_width=3200,
            max_image_height=150,
            max_prediction_length=600,
            channels=1,
        )
    return model

def predict(img,model,filetype):
    if filetype=='line':
        img_path = settings.MEDIA_ROOT+"\images\\"+img
    else:
        img_path=img[:-1]
    print(img_path)
    with open(img_path, 'rb') as img_file:
        img_file_data = img_file.read()
        out, probability = model.predict(img_file_data)
        out_word=""
        i=0
        while i < len(out):
            if out[i:i+2]=='23' or out[i:i+2]=='24':
                out_word = out_word+chr(int(out[i:i+4]))
                i=i+4
            elif out[i:i+2]=='32' or out[i:i+2]=='35' or out[i:i+2]=='95' or out[i:i+2]=='46' or out[i:i+2]=='44' or out[i:i+2]=='45' or (out[i:i+2]<='57' and out[i:i+2]>='48'):
                out_word = out_word+chr(int(out[i:i+2]))
                i = i+2
            elif out[i:i+3]=='124':
                out_word = out_word+chr(int(out[i:i+3]))
                i=i+3
            else:
                break
        print(out_word)
        return out_word
