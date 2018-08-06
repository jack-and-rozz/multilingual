# coding: utf-8 
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.util import nest
from core.models.base import setup_cell
from utils.tf_utils import linear, shape, cnn, flatten

def merge_state(state):
  if isinstance(state[0], LSTMStateTuple):
    new_c = tf.concat([s.c for s in state], axis=-1)
    new_h = tf.concat([s.h for s in state], axis=-1)
    state = LSTMStateTuple(c=new_c, h=new_h)
  else:
    state = tf.concat(state, -1)
  return state

class CNNEncoder(object):
  def __init__(self, config, keep_prob,
               activation=tf.nn.relu, shared_scope=None):
    self.keep_prob = keep_prob
    self.shared_scope = shared_scope
    self.activation = activation

  def __call__(self, *args, **kwargs):
    return self.encode(*args, **kwargs)

  def encode(self, inputs, sequence_length):
    with tf.variable_scope(self.shared_scope or "CNNEncoder"):
      target_rank = 3 # [*, max_sequence_length, hidden_size]
      flattened_inputs, prev_shape = flatten(inputs, target_rank)
      flattened_aggregated_outputs = cnn(flattened_outputs, 
                                        activation=self.activation)
      target_shape = prev_shape[:-2] + [shape(flattened_aggregated_outputs, -1)]
      outputs = tf.reshape(flattened_aggregated_outputs, target_shape)
    outputs = tf.nn.dropout(outputs, self.keep_prob) 
    return outputs, outputs
CharEncoder = CNNEncoder

class WordEncoder(object):
  def __init__(self, keep_prob, activation=tf.nn.relu, shared_scope=None):
    self.keep_prob = keep_prob
    self.shared_scope = shared_scope

  def __call__(self, *args, **kwargs):
    return self.encode(*args, **kwargs)

  def encode(self, inputs):
    outputs = []
    with tf.variable_scope(self.shared_scope or "WordEncoder"):
      outputs = inputs
    outputs = tf.nn.dropout(outputs, self.keep_prob) 
    return outputs

class RNNEncoder(object):
  def __init__(self, config, keep_prob, embeddings=None,
               activation=tf.nn.relu, shared_scope=None):
    self.keep_prob = keep_prob
    self.embeddings = embeddings
    self.activation = activation
    self.shared_scope = shared_scope
    self.output_size = config.output_size
    self.cell_type = config.cell_type
    self.num_layers = config.num_layers
    with tf.variable_scope('fw_cell', reuse=tf.get_variable_scope().reuse):
      self.cell_fw = setup_cell(self.cell_type, self.output_size, 
                                num_layers=self.num_layers, 
                                keep_prob=self.keep_prob)


  def __call__(self, *args, **kwargs):
    return self.encode(*args, **kwargs)

  def reshape_inputs(self, inputs, sequence_length):
    # Lookup embeddings only when the inputs are integers.
    if inputs.dtype == tf.int32:
      assert self.embeddings is not None
      inputs = tf.nn.embedding_lookup(self.embeddings, inputs)
    # flatten the tensor with rank >= 2 to rank 1 tensor.
    sequence_length, _ = flatten(sequence_length, 1)

    # flatten the tensor with rank >= 4 to rank 3 tensor.
    inputs, prev_shape = flatten(inputs, 3) # [*, max_sequence_length, hidden_size]
    output_shape = prev_shape[:-1] + [self.output_size]
    state_shape = prev_shape[:-2] + [self.output_size]
    return inputs, sequence_length, output_shape, state_shape 

  def encode(self, inputs, sequence_length):
    '''
    - inputs: [batch_size, max_sent_len]
    '''
    with tf.variable_scope(self.shared_scope or "RNNEncoder") as scope:
      inputs, sequence_length, output_shape, state_shape = self.reshape_inputs(inputs, sequence_length)
      outputs, state = tf.nn.dynamic_rnn(
        self.cell_fw, inputs,
        sequence_length=sequence_length, dtype=tf.float32, scope=scope)
      outputs = tf.reshape(outputs, output_shape)
      state = tf.reshape(state, state_shape)
    return outputs, state


class BidirectionalRNNEncoder(RNNEncoder):
  def __init__(self, *args, **kwargs):
    RNNEncoder.__init__(self, *args, **kwargs)
    with tf.variable_scope('bw_cell', reuse=tf.get_variable_scope().reuse):
      self.cell_bw = setup_cell(
        self.cell_type, self.output_size, 
        num_layers=self.num_layers, keep_prob=self.keep_prob
      ) 

  def encode(self, inputs, sequence_length=None):
    '''
    - inputs: [batch_size, max_sent_len, hidden_size]
    '''
    with tf.variable_scope(self.shared_scope or "RNNEncoder") as scope:
      inputs, sequence_length, output_shape, state_shape = self.reshape_inputs(inputs, sequence_length)
      outputs, state = tf.nn.bidirectional_dynamic_rnn(
        self.cell_fw, self.cell_bw, inputs,
        sequence_length=sequence_length, dtype=tf.float32, scope=scope)
      with tf.variable_scope("outputs"):
        outputs = tf.concat(outputs, -1)
        outputs = linear(outputs, self.output_size)
        outputs = tf.nn.dropout(outputs, self.keep_prob)

      with tf.variable_scope("state"):
        state = merge_state(state)
        state = linear(state, self.output_size)
      outputs = tf.reshape(outputs, output_shape)
      state = tf.reshape(state, state_shape)
    return outputs, state


class HierarchicalEncoderWrapper(object):
  def __init__(self, encoders):
    '''
    Args:
    - encoders: a tuple of (sent_encoder, context_encoder).
    '''
    self.encoders = encoders

  def encode(self, inputs, sequence_lengths):
    '''
    Args:
    - inputs: [batch_size, max_context_len, max_sentence_len, embedding_size]
    - sequence_lengths: a tuple of (sentence_length, context_length)
    Res:
    - outputs: [batch_size, max_context_len, hidden_size] 
    - state: [batch_size, hidden_size] 
    '''
    assert len(sequence_lengths) == 2
    _, state = self.encoders[0].encode(inputs, sequence_lengths[0]) #[batch_size, max_context_len, hidden_size]
    return self.encoders[1].encode(state, sequence_lengths[1]) # [batch_size, ]
    

    
