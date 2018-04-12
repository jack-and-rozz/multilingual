# coding:utf-8
# https://github.com/tensorflow/tensorflow/issues/11598
import math, sys, time
import numpy as np
from pprint import pprint

import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from utils.tf_utils import shape, linear, make_summary
from utils.common import flatten, timewatch
from core.models.base import ModelBase, setup_cell
from core.models.encoder import CharEncoder, WordEncoder, RNNEncoder, CNNEncoder
from core.models import encoder
from core.extensions.pointer import pointer_decoder 
from core.vocabularies import BOS_ID, PAD_ID, BooleanVocab

# class SharedDenseLayer(tf.layers.Dense):
#   def __init__(self, *args, **kwargs):
#     super(SharedDenseLayer, self).__init__(*args, **kwargs)
#     self.kernel = None

#   def add_kernel(self, tensor):
#     assert shape(tensor, 0) == self.units
#     self.kernel = tensor 

#   def build(self, input_shape):
#     input_shape = tensor_shape.TensorShape(input_shape)
#     if input_shape[-1].value is None:
#       raise ValueError('The last dimension of the inputs to `Dense` '
#                        'should be defined. Found `None`.')
#     self.input_spec = base.InputSpec(min_ndim=2,
#                                      axes={-1: input_shape[-1].value})
#     if not self.kernel:
#       self.kernel = self.add_variable('kernel',
#                                       shape=[input_shape[-1].value, self.units],
#                                       initializer=self.kernel_initializer,
#                                       regularizer=self.kernel_regularizer,
#                                       constraint=self.kernel_constraint,
#                                       dtype=self.dtype,
#                                       trainable=True)
#     if self.use_bias:
#       self.bias = self.add_variable('bias',
#                                     shape=[self.units,],
#                                     initializer=self.bias_initializer,
#                                     regularizer=self.bias_regularizer,
#                                     constraint=self.bias_constraint,
#                                     dtype=self.dtype,
#                                     trainable=True)
#     else:
#       self.bias = None
#     self.built = True

class HierarchicalSeq2Seq(ModelBase):
  def __init__(self, sess, config, w_vocab, c_vocab):
    ModelBase.__init__(self, sess, config)
    self.w_vocab = w_vocab
    self.c_vocab = c_vocab
    with tf.name_scope('Placeholders'):
      self.setup_placeholder(config)

    with tf.variable_scope('Embeddings'):
      self.setup_embeddings(config, w_vocab, c_vocab)

    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
      assert self.w_vocab or self.c_vocab
      word_repls = self.setup_word_encoder(config)

      with tf.variable_scope('Utterance') as scope:
        uttr_repls = self.setup_uttr_encoder(
          config, word_repls, scope=scope)

      with tf.variable_scope('Context') as scope:
        encoder_outputs, encoder_state = self.setup_context_encoder(
          config, uttr_repls, scope=scope)

    # For models that take different decode_state in training and testing.
    with tf.variable_scope('Intermediate') as scope:
      train_decoder_state, test_decode_state, attention_states = self.setup_decoder_states(config, encoder_outputs, encoder_state, scope=scope)

    with tf.variable_scope('Projection') as scope:
      projection_layer = self.setup_projection(config, scope=scope)
    ## Decoder
    with tf.variable_scope('Decoder') as scope:
      self.logits, self.predictions = self.setup_decoder(
        config, train_decoder_state, test_decode_state,
        attention_states=attention_states, 
        encoder_input_lengths=self.context_lengths, 
        projection_layer=projection_layer,
        scope=scope)
      
    with tf.name_scope('Loss'):
      self.loss = self.get_loss(config)

    with tf.name_scope("Update"):
      self.updates = self.get_updates(self.loss)
    self.debug = []
    #self.debug = [self.e_inputs_w_ph, self.d_outputs_ph, decoder_inputs, targets, target_length, target_weights]

  ###################################################
  ##           Components of Network
  ###################################################

  def setup_placeholder(self, config):
    '''
    Prepare tf.placeholder and their lengthes. 
    They are kept as instance variables.
    '''
    self.e_inputs_w_ph = tf.placeholder(
      tf.int32, [None, None, None], name="EncoderInputWords")
    self.e_inputs_c_ph = tf.placeholder(
      tf.int32, [None, None, None, None], name="EncoderInputChars")
    self.d_outputs_ph = tf.placeholder(
      tf.int32, [None, None], name="DecoderOutput")
    self.speaker_changes_ph = tf.placeholder(
      tf.int32, [None, None], name="SpeakerChanges")

    self.is_training = tf.placeholder(tf.bool, [], name='is_training')
    with tf.name_scope('keep_prob'):
      self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate

    with tf.name_scope('batch_size'):
      self.batch_size = batch_size = shape(self.d_outputs_ph, 0)
    with tf.name_scope('start_tokens'):
      self.start_tokens = tf.tile(tf.constant([BOS_ID], dtype=tf.int32), [batch_size])
    with tf.name_scope('end_tokens'):
      self.end_token = PAD_ID
      end_tokens = tf.tile(tf.constant([self.end_token], dtype=tf.int32), [batch_size])
    # Count the length of each dialogue, utterance, (word).
    with tf.name_scope('utterance_length'):
      self.uttr_lengths = tf.count_nonzero(self.e_inputs_w_ph, 
                                           axis=2, dtype=tf.int32)
    with tf.name_scope('context_length'):
      self.context_lengths = tf.count_nonzero(self.uttr_lengths, 
                                              axis=1, dtype=tf.int32)
    '''
    # Example of the decoder's inputs and outputs.
    Against a given input ['how', 'are', 'you', '?'] to the decoder's placeholder,
    - decoder's input : ['_BOS', 'how', 'are', 'you', '?']
    - decoder's output (target) : ['how', 'are', 'you', '?', '_PAD']
    - target_length: 5
    - target_weights: [1, 1, 1, 1, 1]
    Here, the token _PAD behaves as EOS.
    '''
    with tf.name_scope('decoder_inputs'):
      self.decoder_inputs = tf.concat([tf.expand_dims(self.start_tokens, 1), self.d_outputs_ph], axis=1)
    # the length of decoder's inputs/outputs is increased by 1 because of BOS or EOS.
    with tf.name_scope('target_lengths'):
      self.target_length = tf.count_nonzero(self.d_outputs_ph, axis=1, dtype=tf.int32)+1 
    with tf.name_scope('target_weights'):
      self.target_weights = tf.sequence_mask(self.target_length, dtype=tf.float32)

    with tf.name_scope('targets'):
      self.targets = tf.concat([self.d_outputs_ph, tf.expand_dims(end_tokens, 1)], axis=1)[:, :shape(self.target_weights, 1)]

  def setup_embeddings(self, config, w_vocab, c_vocab):
    if w_vocab.embeddings is not None:
      initializer = tf.constant_initializer(w_vocab.embeddings) 
      trainable = config.train_embedding
    else:
      initializer = None
      trainable = True 
    self.w_embeddings = self.initialize_embeddings(
      'Word', [w_vocab.size, config.w_embedding_size],
      initializer=initializer,
      trainable=trainable)

    if c_vocab is not None:
      if c_vocab.embeddings:
        initializer = tf.constant_initializer(c_vocab.embeddings) 
        trainable = config.train_embedding
      else:
        initializer = None
        trainable = True 
      self.c_embeddings = self.initialize_embeddings(
        'Char', [c_vocab.size, config.c_embedding_size],
        initializer=initializer,
        trainable=trainable)
    self.sc_embeddings = self.initialize_embeddings(
      'SpeakerChange', [BooleanVocab.size, config.feature_size],
      trainable=trainable)
    
  def setup_word_encoder(self, config, scope=None):
    word_repls = []
    with tf.variable_scope('Word') as scope:
      w_inputs = tf.nn.embedding_lookup(self.w_embeddings, self.e_inputs_w_ph)
      word_encoder = WordEncoder(self.keep_prob, shared_scope=scope)
      word_repls.append(word_encoder.encode(w_inputs))

    with tf.variable_scope('Char') as scope:
      if self.c_vocab:
        c_inputs = tf.nn.embedding_lookup(self.c_embeddings, self.e_inputs_c_ph)
        char_encoder = CNNEncoder(self.keep_prob, shared_scope=scope)
        word_repls.append(char_encoder.encode(c_inputs))
    word_repls = tf.concat(word_repls, axis=-1) # [batch_size, context_len, utterance_len, word_emb_size + cnn_output_size]
    return word_repls

  def setup_uttr_encoder(self, config, word_repls, scope=None):
    encoder_type = getattr(encoder, config.encoder.utterance.encoder_type)
    self.uttr_encoder = encoder_type(config.encoder.utterance, 
                                     self.keep_prob, 
                                     shared_scope=scope)
    uttr_repls, _ = self.uttr_encoder.encode(word_repls, self.uttr_lengths)
    if len(uttr_repls.get_shape()) == 4:
      '''
      Use the last state of each utterance encoded by RNN.
      [batch_size, context_len, uttr_len, hidden_size]
      -> [batch_size, context_len, hidden_size] 
      '''
      uttr_repls = uttr_repls[:, :, -1, :] 
    # Concatenate the feature_embeddings with each utterance representations.
    speaker_changes = tf.nn.embedding_lookup(self.sc_embeddings, 
                                             self.speaker_changes_ph)
    uttr_repls = tf.concat([uttr_repls, speaker_changes], axis=-1)
    return uttr_repls

  def setup_context_encoder(self, config, uttr_repls, scope=None):
    encoder_type = getattr(encoder, config.encoder.context.encoder_type)
    self.context_encoder = encoder_type(config.encoder.context, 
                                        self.keep_prob, 
                                        shared_scope=scope)
    return self.context_encoder.encode(uttr_repls, self.context_lengths)

  def setup_decoder_states(self, config, encoder_outputs, encoder_state, 
                           scope=None):
    attention_states = encoder_outputs
    decoder_state = encoder_state
    return decoder_state, decoder_state, attention_states

  def setup_projection(self, config, scope=None):
    return None

  def setup_decoder(self, config, train_decoder_state, test_decoder_state,
                    encoder_input_lengths=None,
                    attention_states=None, 
                    projection_layer=None,
                    scope=None):
    batch_size = self.batch_size
    decoder_inputs_emb = tf.nn.embedding_lookup(self.w_embeddings, self.decoder_inputs)
    # TODO: 多言語対応にする時はbias, trainableをfalseにしてembeddingをconstantにしたい
    decoder_cell = setup_cell(config.decoder.cell_type, 
                              shape(train_decoder_state, -1), 
                              config.decoder.num_layers,
                              keep_prob=self.keep_prob)
    if projection_layer is None:
      with tf.variable_scope('projection') as scope:
        projection_layer = tf.layers.Dense(config.w_vocab_size, use_bias=True, trainable=True)

    with tf.name_scope('Training'):
      if config.attention_type:
        assert attention_states is not None
        num_units = shape(attention_states, -1)
        attention = tf.contrib.seq2seq.LuongAttention(
          num_units, attention_states,
          memory_sequence_length=encoder_input_length)
        train_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
          decoder_cell, attention)
        decoder_initial_state = train_decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=train_decoder_state)
      else:
        train_decoder_cell = decoder_cell
        decoder_initial_state = train_decoder_state

      # encoder_state can't be directly copied into decoder_cell when using the attention mechanisms, initial_state must be an instance of AttentionWrapperState. (https://github.com/tensorflow/nmt/issues/205)

      helper = tf.contrib.seq2seq.TrainingHelper(
        decoder_inputs_emb, sequence_length=self.target_length, time_major=False)

      decoder = tf.contrib.seq2seq.BasicDecoder(
        train_decoder_cell, helper, decoder_initial_state,
        output_layer=projection_layer)
      train_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder, 
        impute_finished=True,
        maximum_iterations=tf.reduce_max(self.target_length),
        scope=scope)
      logits = train_decoder_outputs.rnn_output

    with tf.name_scope('Test'):
      beam_width = config.beam_width
      if config.attention_type:
        num_units = shape(attention_states, -1)
        attention = tf.contrib.seq2seq.LuongAttention(
          num_units, 
          tf.contrib.seq2seq.tile_batch(
            attention_states, multiplier=beam_width),
          memory_sequence_length=tf.contrib.seq2seq.tile_batch(
            encoder_input_length, multiplier=beam_width))
        test_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
          decoder_cell, attention)
        decoder_initial_state = test_decoder_cell.zero_state(batch_size*beam_width, tf.float32).clone(cell_state=tf.contrib.seq2seq.tile_batch(test_decoder_state, multiplier=beam_width))
      else:
        test_decoder_cell = decoder_cell
        decoder_initial_state = tf.contrib.seq2seq.tile_batch(
          test_decoder_state, multiplier=beam_width)

      decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        test_decoder_cell, self.w_embeddings, self.start_tokens, self.end_token, 
        decoder_initial_state,
        beam_width, output_layer=projection_layer,
        length_penalty_weight=config.length_penalty_weight)
      test_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder, impute_finished=False,
        maximum_iterations=config.utterance_max_len, scope=scope)
      predictions = test_decoder_outputs.predicted_ids
    return logits, predictions

  def get_loss(self, config):
    self.divergence = tf.constant(0)
    self.crossent = tf.contrib.seq2seq.sequence_loss(
      self.logits, self.targets, self.target_weights,
      average_across_timesteps=True, average_across_batch=True)
    return self.crossent

  ###################################################
  ##           Training / Testing
  ###################################################

  def get_input_feed(self, batch, is_training):
    feed_dict = {
      self.d_outputs_ph: np.array(batch.responses),
      self.is_training: is_training,
      self.speaker_changes_ph: np.array(batch.speaker_changes)
    }
    feed_dict[self.e_inputs_w_ph] = np.array(batch.w_contexts)
    if self.c_vocab:
      feed_dict[self.e_inputs_c_ph] = np.array(batch.c_contexts)
    # print '<<<<<get_input_feed>>>>'
    # for k,v in feed_dict.items():
    #   if type(v) == np.ndarray:
    #     print k, v #v.shape

    return feed_dict

  @timewatch()
  def train(self, data, do_update=True):
    '''
    This method can be used for the calculation of valid loss with do_update=False
    '''
    loss = 0.0
    divergence = 0.0
    num_steps = 0
    epoch_time = 0.0
    for i, batch in enumerate(data):
      feed_dict = self.get_input_feed(batch, do_update)
      # for x,resx in zip(self.debug, self.sess.run(self.debug, feed_dict)):
      #   print x
      #   print resx, resx.shape
      # exit(1)
      t = time.time()
      output_feed = [self.crossent, self.divergence] 
      if do_update:
        output_feed.append(self.updates)
      res = self.sess.run(output_feed, feed_dict)
      step_loss = math.exp(res[0])
      divergence += res[1]
      print self.epoch.eval(), i, step_loss
      print 'res', res
      sys.stdout.flush()
      if math.isnan(step_loss):
        sys.stderr.write('Got a Nan loss.\n')
        #for x in feed_dict:
        #  print x
        #  print feed_dict[x]
        #exit(1)
      epoch_time += time.time() - t
      loss += step_loss
      num_steps += 1
    loss /= num_steps
    divergence /= num_steps
    mode = 'train' if do_update else 'valid'
    summary = make_summary({
      "%s/cross_entropy" % mode : loss,
      "%s/divergence" % mode : divergence,
    })
    return loss, summary, epoch_time

  @timewatch()
  def test(self, data):
    inputs = []
    outputs = []
    speaker_changes = []
    predictions = []
    num_steps = 0
    epoch_time = 0.0
    for i, batch in enumerate(data):
      feed_dict = self.get_input_feed(batch, False)
      # for x,resx in zip(self.debug, self.sess.run(self.debug, feed_dict)):
      #    print x
      #    print resx.shape
      # exit(1)
      t = time.time()
      batch_predictions = self.sess.run(self.predictions, feed_dict)
      epoch_time += time.time() - t
      num_steps += 1
      inputs.append(batch.w_contexts)
      outputs.append(batch.responses)
      speaker_changes.append(batch.speaker_changes)
      predictions.append(batch_predictions)
    inputs = flatten(inputs)
    outputs = flatten(outputs)
    speaker_changes = flatten(speaker_changes)
    predictions = flatten(predictions)
    inputs = [[self.w_vocab.id2sent(u, join=True) for u in c] for c in inputs]
    outputs = [self.w_vocab.id2sent(r, join=True) for r in outputs]
    # [batch_size, utterance_max_len, beam_width] - > [batch_size, beam_width, utterance_max_len]
    predictions = [[self.w_vocab.id2sent(r, join=True) for r in zip(*p)] for p in predictions]
    speaker_changes = [BooleanVocab.id2sent(sc) for sc in speaker_changes]
    return (inputs, outputs, speaker_changes, predictions), epoch_time


class VariationalHierarchicalSeq2Seq(HierarchicalSeq2Seq):
  def setup_decoder_states(self, config, encoder_outputs, encoder_state,
                           scope=None):
    attention_states = encoder_outputs

    response_emb = tf.nn.embedding_lookup(self.w_embeddings, self.d_outputs_ph)
    response_lengths = tf.count_nonzero(self.d_outputs_ph, 
                                        axis=1, dtype=tf.int32)
    print 'encoder_state', encoder_state
    print 'encoder_outputs', encoder_outputs

    _, h_future = self.uttr_encoder.encode(response_emb, response_lengths)
    print 'h_future', h_future
    def _get_distribution(state, output_size):
      h = state
      num_layers = 1 
      for i in range(num_layers):
        with tf.variable_scope('linear%d' % i) as scope:
          h = linear(h, output_size, scope=scope)
      with tf.variable_scope('Mean'):
        mean = linear(h, output_size, activation=None)
      with tf.variable_scope('Var'):
        var = linear(h, output_size, activation=tf.nn.softplus)
      return tfd.MultivariateNormalDiag(mean, var)

    output_size = shape(encoder_state, -1)
    with tf.variable_scope('Prior'):
      self.prior = _get_distribution(encoder_state, output_size)
    with tf.variable_scope('Posterior'):
      self.posterior = _get_distribution(
        tf.concat([encoder_state, h_future], axis=-1),
        output_size)

    train_decoder_state = tf.concat([encoder_state, self.posterior.sample()], axis=-1)
    test_decoder_state = tf.concat([encoder_state, self.prior.sample()], axis=-1)
    #train_decoder_state = encoder_state + self.posterior.sample()
    #test_decoder_state = encoder_state + self.prior.sample()
    print train_decoder_state
    print test_decoder_state
    return train_decoder_state, test_decoder_state, attention_states

  def get_loss(self, config):
    self.divergence = tf.reduce_mean(tfd.kl_divergence(self.posterior, self.prior))
    self.crossent = tf.contrib.seq2seq.sequence_loss(
      self.logits, self.targets, self.target_weights,
      average_across_timesteps=True, average_across_batch=True)
    loss = self.divergence + self.crossent
    #print self.divergence
    #print self.crossent
    #print loss
    #exit(1)
    return loss
