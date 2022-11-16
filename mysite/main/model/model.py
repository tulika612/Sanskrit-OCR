"""Visual Attention Based OCR Model."""

from __future__ import absolute_import
from __future__ import division
from django.conf import settings



import time
import os
import math
import logging
import sys

import distance
import numpy as np
import tensorflow as tf

from six.moves import xrange  # pylint: disable=redefined-builtin
from .cnn import CNN
from .seq2seq_model import Seq2SeqModel
from .util.data_gen import DataGen
from .util.visualizations import visualize_attention


class Model(object):
    def __init__(self,
                 phase,
                 visualize,
                 output_dir,
                 batch_size,
                 initial_learning_rate,
                 steps_per_checkpoint,
                 model_dir,
                 target_embedding_size,
                 attn_num_hidden,
                 attn_num_layers,
                 clip_gradients,
                 max_gradient_norm,
                 session,
                 load_model,
                 gpu_id,
                 use_gru,
                 use_distance=True,
                 max_image_width=160,
                 max_image_height=60,
                 max_prediction_length=50,
                 channels=1,
                 reg_val=0):

        self.use_distance = use_distance

        # We need resized width, not the actual width
        max_resized_width = 1. * max_image_width / max_image_height * DataGen.IMAGE_HEIGHT

        self.max_original_width = max_image_width
        self.max_width = int(math.ceil(max_resized_width))
        self.max_label_length = max_prediction_length
        self.encoder_size = int(math.ceil(1. * self.max_width / 4))
        self.decoder_size = max_prediction_length + 2
        self.buckets = [(self.encoder_size, self.decoder_size)]

        if gpu_id >= 0:
            device_id = '/gpu:' + str(gpu_id)
        else:
            device_id = '/cpu:0'
        self.device_id = device_id

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if phase == 'test':
            batch_size = 1

        logging.info('phase: %s', phase)
        logging.info('model_dir: %s', model_dir)
        logging.info('load_model: %s', load_model)
        logging.info('output_dir: %s', output_dir)
        logging.info('steps_per_checkpoint: %d', steps_per_checkpoint)
        logging.info('batch_size: %d', batch_size)
        logging.info('learning_rate: %f', initial_learning_rate)
        logging.info('reg_val: %d', reg_val)
        logging.info('max_gradient_norm: %f', max_gradient_norm)
        logging.info('clip_gradients: %s', clip_gradients)
        logging.info('max_image_width %f', max_image_width)
        logging.info('max_prediction_length %f', max_prediction_length)
        logging.info('channels: %d', channels)
        logging.info('target_embedding_size: %f', target_embedding_size)
        logging.info('attn_num_hidden: %d', attn_num_hidden)
        logging.info('attn_num_layers: %d', attn_num_layers)
        logging.info('visualize: %s', visualize)

        if use_gru:
            logging.info('using GRU in the decoder.')

        self.reg_val = reg_val
        self.sess = session
        self.steps_per_checkpoint = steps_per_checkpoint
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_label_lengthc =int(self.max_label_length/4)
        self.global_step = tf.Variable(0, trainable=False)
        self.phase = phase
        self.visualize = visualize
        self.learning_rate = initial_learning_rate
        self.clip_gradients = clip_gradients
        self.channels = channels

        if phase == 'train':
            self.forward_only = False
        else:
            self.forward_only = True

        with tf.device(device_id):

            self.height = tf.constant(DataGen.IMAGE_HEIGHT, dtype=tf.int32)
            self.height_float = tf.constant(DataGen.IMAGE_HEIGHT, dtype=tf.float64)

            self.img_pl = tf.placeholder(tf.string, name='input_image_as_bytes')
            self.labels = tf.placeholder(tf.int32,shape=(self.batch_size, self.max_label_lengthc), name="input_labels_as_bytes")
            #self.label_data = tf.placeholder(tf.string, shape=[None,self.max_label_length], name="input_labels_as_bs")
            self.img_data = tf.cond(
                tf.less(tf.rank(self.img_pl), 1),
                lambda: tf.expand_dims(self.img_pl, 0),
                lambda: self.img_pl
            )
            self.img_data = tf.map_fn(self._prepare_image, self.img_data, dtype=tf.float32)
            num_images = tf.shape(self.img_data)[0]

            # TODO: create a mask depending on the image/batch size
            self.encoder_masks = []
            for i in xrange(self.encoder_size + 1):
                self.encoder_masks.append(
                    tf.tile([[1.]], [num_images, 1])
                )

            self.decoder_inputs = []
            self.target_weights = []
            for i in xrange(self.decoder_size + 1):
                self.decoder_inputs.append(
                    tf.tile([1], [num_images])
                )
                if i < self.decoder_size:
                    self.target_weights.append(tf.tile([1.], [num_images]))
                else:
                    self.target_weights.append(tf.tile([0.], [num_images]))

            cnn_model = CNN(self.img_data, not self.forward_only)
            self.conv_output = cnn_model.tf_output()
            self.perm_conv_output = tf.transpose(self.conv_output, perm=[1, 0, 2])
            self.attention_decoder_model = Seq2SeqModel(
                encoder_masks=self.encoder_masks,
                encoder_inputs_tensor=self.perm_conv_output,
                labels=self.labels,
                decoder_inputs=self.decoder_inputs,
                target_weights=self.target_weights,
                batch_size = self.batch_size,
                target_vocab_size=len(DataGen.CHARMAP),
                buckets=self.buckets,
                target_embedding_size=target_embedding_size,
                attn_num_layers=attn_num_layers,
                attn_num_hidden=attn_num_hidden,
                forward_only=self.forward_only,
                use_gru=use_gru)

            table = tf.contrib.lookup.MutableHashTable(
                key_dtype=tf.int64,
                value_dtype=tf.string,
                default_value="",
                checkpoint=True,
            )

            insert = table.insert(
                tf.constant(list(range(len(DataGen.CHARMAP))), dtype=tf.int64),
                tf.constant(DataGen.CHARMAP),
            )

            with tf.control_dependencies([insert]):
                num_feed = []
                prb_feed = []

                for line in xrange(len(self.attention_decoder_model.output)):
                    guess = tf.argmax(self.attention_decoder_model.output[line], axis=1)
                    proba = tf.reduce_max(
                        tf.nn.softmax(self.attention_decoder_model.output[line]), axis=1)
                    num_feed.append(guess)
                    prb_feed.append(proba)

                # Join the predictions into a single output string.
                trans_output = tf.transpose(num_feed)
                trans_output = tf.map_fn(
                    lambda m: tf.foldr(
                        lambda a, x: tf.cond(
                            tf.equal(x, DataGen.EOS_ID),
                            lambda: '',
                            lambda: table.lookup(x) + a  # pylint: disable=undefined-variable
                        ),
                        m,
                        initializer=''
                    ),
                    trans_output,
                    dtype=tf.string
                )

                # Calculate the total probability of the output string.
                trans_outprb = tf.transpose(prb_feed)
                trans_outprb = tf.gather(trans_outprb, tf.range(tf.size(trans_output)))
                trans_outprb = tf.map_fn(
                    lambda m: tf.foldr(
                        lambda a, x: tf.multiply(tf.cast(x, tf.float64), a),
                        m,
                        initializer=tf.cast(1, tf.float64)
                    ),
                    trans_outprb,
                    dtype=tf.float64
                )

                self.prediction = tf.cond(
                    tf.equal(tf.shape(trans_output)[0], 1),
                    lambda: trans_output[0],
                    lambda: trans_output,
                )
                self.probability = tf.cond(
                    tf.equal(tf.shape(trans_outprb)[0], 1),
                    lambda: trans_outprb[0],
                    lambda: trans_outprb,
                )

                self.prediction = tf.identity(self.prediction, name='prediction')
                self.probability = tf.identity(self.probability, name='probability')

        self.saver_all = tf.train.Saver(tf.all_variables())
        self.checkpoint_path =settings.BASE_DIR+self.model_dir+'model.ckpt-310100'
        print(self.checkpoint_path)

        #ckpt = tf.train.get_checkpoint_state(model_dir)
        #print(ckpt.model_checkpoint_path)
        if load_model:
            # pylint: disable=no-member
            logging.info("Reading model parameters from %s", self.checkpoint_path)
            self.saver_all.restore(self.sess, self.checkpoint_path)
        else:
            logging.info("Created model with fresh parameters.")
            self.sess.run(tf.initialize_all_variables())

    def predict(self, image_file_data):
        input_feed = {}
        input_feed[self.img_pl.name] = image_file_data

        output_feed = [self.prediction, self.probability]
        outputs = self.sess.run(output_feed, input_feed)

        text = outputs[0]
        probability = outputs[1]
        if sys.version_info >= (3,):
            text = text.decode('iso-8859-1')

        return (text, probability)

    def _prepare_image(self, image):
        """Resize the image to a maximum height of `self.height` and maximum
        width of `self.width` while maintaining the aspect ratio. Pad the
        resized image to a fixed size of ``[self.height, self.width]``."""
        img = tf.image.decode_png(image, channels=self.channels)
        dims = tf.shape(img)
        width = self.max_width

        max_width = tf.to_int32(tf.ceil(tf.truediv(dims[1], dims[0]) * self.height_float))
        max_height = tf.to_int32(tf.ceil(tf.truediv(width, max_width) * self.height_float))

        resized = tf.cond(
            tf.greater_equal(width, max_width),
            lambda: tf.cond(
                tf.less_equal(dims[0], self.height),
                lambda: tf.to_float(img),
                lambda: tf.image.resize_images(img, [self.height, max_width],
                                               method=tf.image.ResizeMethod.BICUBIC),
            ),
            lambda: tf.image.resize_images(img, [max_height, width],
                                           method=tf.image.ResizeMethod.BICUBIC)
        )

        padded = tf.image.pad_to_bounding_box(resized, 0, 0, self.height, width)
        return padded
