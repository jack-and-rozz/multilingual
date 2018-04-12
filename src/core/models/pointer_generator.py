# coding:utf-8
import math, sys, time
import numpy as np
from pprint import pprint

import tensorflow as tf
from utils.tf_utils import shape
from core.models import ModelBase, setup_cell
from core.models.encoder import WordEncoder, SentenceEncoder
from core.extensions.pointer import pointer_decoder 
from core.vocabularies import BOS_ID

class PointerGeneratorNetwork(ModelBase):
  def __init__(self, sess, conf, vocab):
    ModelBase.__init__(self, sess, conf)
    self.vocab = vocab
    input_max_len, output_max_len = None, conf.output_max_len
    self.is_training = tf.placeholder(tf.bool, [], name='is_training')
    with tf.name_scope('keep_prob'):
      self.keep_prob = 1.0 - tf.to_float(self.is_training) * conf.dropout_rate

    with tf.name_scope('EncoderInput'):
      self.e_inputs_ph = tf.placeholder(
        tf.int32, [None, input_max_len], name="EncoderInput")

    with tf.name_scope('batch_size'):
      batch_size = shape(self.e_inputs_ph, 0)

    with tf.variable_scope('Embeddings') as scope:
      self.w_embeddings = self.initialize_embeddings(
        'Word', vocab.embeddings.shape, 
        initializer=tf.constant_initializer(vocab.embeddings),
        trainable=conf.train_embedding)

