# coding:utf-8
# https://github.com/tensorflow/tensorflow/issues/11598
import math, sys, time
import numpy as np
from pprint import pprint

import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from utils.tf_utils import shape, linear, make_summary
from utils.common import flatten, timewatch, dotDict, recDotDict
from core.models.base import ModelBase, setup_cell
from core.models.encoder import CharEncoder, WordEncoder, RNNEncoder, CNNEncoder, HierarchicalEncoderWrapper
from core.models.decoder import get_start_and_end_tokens, SentenceDecoder
from core.models import encoder
from core.extensions.pointer import pointer_decoder 
from core.vocabularies import EOS_ID, BOS_ID, PAD_ID, BooleanVocab

#from core.models.autoencoder import AutoEncoder
#from core.models.seq2seq import HierarchicalSeq2Seq as Dialogue

class Seq2Seq(ModelBase):
  def __init__(self, sess, config, encoder, decoder, 
               domain_feature, shared_scope=None):
    self.inputs, _, enc_state = self.setup_encoding(encoder) # enc_state: [batch_size, hidden_size]
    with tf.variable('Intermediate'):
      domain_feature = tf.tile(domain_feature, [shape(enc_state, 0), 1])
      decoder_rnn_size = shape(enc_state, -1)
      enc_state = tf.concat([enc_state, domain_feature], axis=-1)
      enc_state = linear(enc_state, decoder_rnn_size)

    self.outputs, self.crossent, self.predictions = self.setup_decoding(decoder, enc_state)
    self.loss = self.crossent

  def setup_encoding(self, encoder):
    # Setup Placeholders.
    inputs = tf.placeholder(
      tf.int32, [None, None], name="InputPlaceholder")
    # Setup encoder's input information.
    with tf.name_scope('input_context_length'):
      sent_length = tf.count_nonzero(inputs, axis=1, dtype=tf.int32)

    # Encoding.
    enc_outputs, enc_state = encoder.encode(inputs, sent_length)
    return inputs, enc_outputs, enc_state

  def setup_decoding(self, decoder, enc_state):
    outputs_ph = tf.placeholder(
      tf.int32, [None, None], name="OutputPlaceholder")
    # Setup decoder's input and output, and their information.
    with tf.name_scope('batch_size'):
      batch_size = shape(outputs_ph, 0)

    _, start_tokens, end_token, end_tokens = get_start_and_end_tokens(batch_size)

    with tf.name_scope('start_tokens'):
      start_tokens = tf.tile(tf.constant([BOS_ID], dtype=tf.int32), [batch_size])
    with tf.name_scope('decoder_inputs'):
      decoder_inputs = tf.concat([tf.expand_dims(start_tokens, 1), outputs_ph], axis=1)

    # The length of decoder's inputs/outputs is increased by 1 because of the prepended BOS or the appended EOS.

    with tf.name_scope('end_tokens'):
      end_token = PAD_ID
      end_tokens = tf.tile(tf.constant([end_token], dtype=tf.int32), [batch_size])
    with tf.name_scope('target_length'):
      target_length = tf.count_nonzero(outputs_ph, axis=1, dtype=tf.int32)+1 
    with tf.name_scope('target_weights'):
      target_weights = tf.sequence_mask(target_length, dtype=tf.float32) # [batch_size, actual_max_sentence_len]
    with tf.name_scope('targets'):
      targets = tf.concat([outputs_ph, tf.expand_dims(end_tokens, 1)], axis=1)
      targets = targets[:, :shape(target_weights, 1)]

    # Decoding.
    logits, predictions = decoder.decode(decoder_inputs, enc_state, target_length)

    crossent = tf.contrib.seq2seq.sequence_loss(
      logits, targets, target_weights,
      average_across_timesteps=True, average_across_batch=True)

    return outputs_ph, crossent, predictions

class HierarchicalSeq2Seq(Seq2Seq):
  def setup_encoding(self, encoder):
    # Setup Placeholders.
    inputs = tf.placeholder(
      tf.int32, [None, None, None], name="InputPlaceholder")

    # Setup encoder's input information.
    with tf.name_scope('input_sentence_length'):
      sent_length = tf.count_nonzero(inputs, axis=2, dtype=tf.int32)
    with tf.name_scope('input_context_length'):
      context_length = tf.count_nonzero(sent_length, axis=1, dtype=tf.int32)
    # Encoding.
    enc_outputs, enc_state = encoder.encode(inputs, (sent_length, context_length))
    return inputs, enc_outputs, enc_state

AutoEncoder = Seq2Seq
HierarchicalDialogueModel = HierarchicalSeq2Seq

class MultiLangDialogueModelWithAutoEncoder(ModelBase):
  def __init__(self, sess, config, vocab):
    ModelBase.__init__(self, sess, config)

    # Setup Placeholders.
    with tf.name_scope('Placeholders'):
      # 0: en, i: ja
      self.source_lang = tf.placeholder(tf.int32, [], name="SourceLang")
      self.target_lang = tf.placeholder(tf.int32, [], name="TargetLang")
      self.task = tf.placeholder(tf.int32, [], name="Task")

    # Initialize embeddings.
    with tf.variable_scope('Embeddings'):
      self.embeddings = recDotDict({
        'word': {}, 'lang': {}, 'task': {},
      })
      trainable=True
      with tf.variable_scope('En'):
        self.embeddings.word.en = self.initialize_embeddings(
          'Word', [vocab.en.word.vocab_size, vocab.en.word.emb_size],
          initializer=tf.constant_initializer(vocab.en.word.embeddings),
          trainable=trainable)
      with tf.variable_scope('Ja'):
        self.embeddings.word.ja = self.initialize_embeddings(
          'Word', [vocab.ja.word.vocab_size, vocab.ja.word.emb_size],
          initializer=tf.constant_initializer(vocab.ja.word.embeddings),
          trainable=trainable)
      self.embeddings.lang = self.initialize_embeddings(
        'Lang', [2, config.lang_embedding_size], trainable=True) # En or Ja
      self.embeddings.task = self.initialize_embeddings(
        'Task', [2, config.task_embedding_size], trainable=True) # DLG or AE

    with tf.variable_scope('Projection'):
      self.projection = recDotDict()
      # self.projection.en = tf.layers.Dense(vocab.en.word.vocab_size, 
      #                                      use_bias=True, trainable=True)
      # self.projection.ja = tf.layers.Dense(vocab.ja.word.vocab_size, 
      #                                      use_bias=True, trainable=True)

      emb_size = vocab.en.word.emb_size if vocab.en.word.emb_size else config.w_embedding_size
      self.projection.en = tf.get_variable('En', shape=[emb_size, vocab.en.word.vocab_size], trainable=True, dtype=tf.float32)

      emb_size = vocab.ja.word.emb_size if vocab.ja.word.emb_size else config.w_embedding_size
      self.projection.ja = tf.get_variable('Ja', shape=[emb_size, vocab.ja.word.vocab_size], trainable=True, dtype=tf.float32)

    source_emb = tf.cond(tf.cast(self.source_lang, tf.bool), 
                         lambda: self.embeddings.word.en, 
                         lambda: self.embeddings.word.ja)
    target_emb = tf.cond(tf.cast(self.target_lang, tf.bool),
                         lambda: self.embeddings.word.en, 
                         lambda: self.embeddings.word.ja)
    projection_layer = tf.cond(tf.cast(self.target_lang, tf.bool),
                               lambda: self.projection.en, 
                               lambda: self.projection.ja)

    with tf.variable_scope('Encoder'):
      with tf.variable_scope('Sent') as scope:
        encoder_type = getattr(encoder, config.encoder.utterance.encoder_type)
        sent_encoder = encoder_type(config.encoder.utterance, self.keep_prob,
                                    embeddings=source_emb, shared_scope=scope)
 
      with tf.variable_scope('Paragraph') as scope:
        encoder_type = getattr(encoder, config.encoder.context.encoder_type)
        context_encoder = encoder_type(config.encoder.context, self.keep_prob, 
                                       embeddings=None, shared_scope=scope)
        context_encoder = HierarchicalEncoderWrapper((sent_encoder, context_encoder))
    with tf.variable_scope('Decoder') as scope:
      decoder = SentenceDecoder(config.decoder, self.keep_prob, target_emb, 
                                projection_layer, scope=scope)
    with tf.name_scope('DomainFeature'):
      domain_feature = tf.concat([
        tf.nn.embedding_lookup(self.embeddings.task, self.task),
        tf.nn.embedding_lookup(self.embeddings.lang, self.target_lang),
      ], axis=-1)
      domain_feature = tf.expand_dims(domain_feature, axis=0)
    shared_scope = tf.get_variable_scope()

    # Define models.
    self.models = recDotDict()
    with tf.name_scope('Dialogue'):
      self.models.dialogue = HierarchicalDialogueModel(
        sess, config, context_encoder, decoder, domain_feature, 
        shared_scope=shared_scope)
    with tf.name_scope('AutoEncoder'):
      self.models.autoencoder = AutoEncoder(
        sess, config, sent_encoder, decoder, domain_feature, 
        shared_scope=shared_scope)

  def train(self, data):
    return self.calc_loss(data, do_update=True)

  def valid(self, data):
    return self.calc_loss(data, do_update=False)

  def calc_loss(self, data, do_update=True):
    exit(1)
    return
