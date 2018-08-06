#coding:utf-8
import math, sys, time
import numpy as np
from pprint import pprint
from core.models.base import ModelBase, setup_cell
from utils.tf_utils import shape, linear, make_summary, SharedKernelDense
import tensorflow as tf
from core.vocabularies import EOS_ID, BOS_ID, PAD_ID, BooleanVocab

def get_start_and_end_tokens(batch_size):
  '''
  Args:
  - batch_size: dynamic batch_size obtained by shape(*, 0).
  Res:
  A token and tiled tokens for start end end of decoding.
  '''
  with tf.name_scope('start_tokens'):
    start_token = BOS_ID
    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size])
    
  with tf.name_scope('end_tokens'):
    end_token = PAD_ID
    end_tokens = tf.tile(tf.constant([end_token], dtype=tf.int32), [batch_size])
  return start_token, start_tokens, end_token, end_tokens

class SentenceDecoder(object):
  def __init__(self, config, keep_prob, 
               embeddings, projection_kernel, scope=None):
    self.cell_type = config.cell_type
    self.num_layers = config.num_layers
    self.hidden_size = config.hidden_size
    self.beam_width = config.beam_width
    self.length_penalty_weight = config.length_penalty_weight
    self.max_len = config.max_len
    self.keep_prob = keep_prob
    self.embeddings = embeddings
    #self.projection_layer = projection_layer
    self.projection_layer = SharedKernelDense(shape(embeddings, 0), 
                                              use_bias=False, trainable=False,
                                              shared_kernel=projection_kernel)
    self.scope=scope
    with tf.variable_scope(self.scope or "SentenceDecoder") as scope:
      self.decoder_cell = setup_cell(self.cell_type, self.hidden_size, 
                                     self.num_layers, keep_prob=self.keep_prob)

    

  def decode(self, inputs, initial_state, sequence_length):
    decoder_inputs_emb = tf.nn.embedding_lookup(self.embeddings, inputs)
    with tf.name_scope('batch_size'):
      batch_size = shape(initial_state, 0)
    _, start_tokens, end_token, _ = get_start_and_end_tokens(batch_size)

    scope = tf.get_variable_scope()
    with tf.name_scope('Training'):
      train_decoder_cell = self.decoder_cell
      train_decoder_state = initial_state
      helper = tf.contrib.seq2seq.TrainingHelper(
        decoder_inputs_emb, sequence_length=sequence_length, time_major=False)

      decoder = tf.contrib.seq2seq.BasicDecoder(
        train_decoder_cell, helper, train_decoder_state,
        output_layer=self.projection_layer)
      train_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder, 
        impute_finished=True,
        maximum_iterations=tf.reduce_max(sequence_length),
        scope=scope)
      logits = train_decoder_outputs.rnn_output

    with tf.name_scope('Test'):
      test_decoder_cell = self.decoder_cell
      test_decoder_state = tf.contrib.seq2seq.tile_batch(
        initial_state, multiplier=self.beam_width)
      decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        test_decoder_cell, self.embeddings, start_tokens, end_token, 
        test_decoder_state,
        self.beam_width, output_layer=self.projection_layer,
        length_penalty_weight=self.length_penalty_weight)
      test_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder, impute_finished=False,
        maximum_iterations=self.max_len, scope=scope)
      predictions = test_decoder_outputs.predicted_ids
    return logits, predictions
