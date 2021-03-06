# coding:utf-8
import math, sys, time
import numpy as np
from pprint import pprint

import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from utils.tf_utils import shape, linear, make_summary, SharedKernelDense
from utils.common import flatten, timewatch, dotDict
from core.models.base import ModelBase, setup_cell
from core.models.encoder import CharEncoder, WordEncoder, RNNEncoder, CNNEncoder
from core.models import encoder
from core.extensions.pointer import pointer_decoder 
from core.vocabularies import BOS_ID, PAD_ID, BooleanVocab


class AutoEncoder(ModelBase):
  def __init__(self, sess, config, vocab):
    ModelBase.__init__(self, sess, config)
    self.vocab = vocab
    with tf.name_scope('Placeholders'):
      self.setup_placeholder(config)

    with tf.variable_scope('Embeddings'):
      self.setup_embeddings(config, vocab)

    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
      word_repls = self.setup_word_encoder(config)

      with tf.variable_scope('Utterance') as scope:
        encoder_outputs, encoder_state = self.setup_uttr_encoder(
          config, word_repls, scope=scope)

    # For models that take different decode_state in training and testing.
    with tf.variable_scope('Intermediate') as scope:
      train_decoder_state, test_decode_state, attention_states = self.setup_decoder_states(config, encoder_outputs, encoder_state, scope=scope)

    with tf.variable_scope('Projection') as scope:
      projection_layer = self.setup_projection(config, scope=scope)
    ## Decoder
    with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE) as scope:
      self.logits, self.e_predictions = self.setup_decoder(
        config, train_decoder_state, test_decode_state,
        self.embeddings.e_word,
        attention_states=attention_states, 
        encoder_input_lengths=self.uttr_lengths, 
        projection_layer=projection_layer,
        scope=scope)

      _, self.j_predictions = self.setup_decoder(
        config, train_decoder_state, test_decode_state,
        self.embeddings.j_word,
        attention_states=attention_states, 
        encoder_input_lengths=self.uttr_lengths, 
        projection_layer=projection_layer,
        scope=scope)

    with tf.name_scope('Loss'):
      self.loss = self.get_loss(config)

    with tf.name_scope("Update"):
      self.updates = self.get_updates(self.loss)
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
      tf.int32, [None, None], name="EncoderInputWords")
    self.e_inputs_c_ph = tf.placeholder(
      tf.int32, [None, None, None], name="EncoderInputChars")
    #self.d_outputs_ph = tf.placeholder(
    #  tf.int32, [None, None], name="DecoderOutput")
    self.d_outputs_ph = self.e_inputs_w_ph

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

  def setup_embeddings(self, config, vocab):
    self.embeddings = dotDict()
    n_start_vocab = len(vocab.e_word.start_vocab)
    special_tokens_emb = self.initialize_embeddings(
      'SpecialTokens', vocab.e_word.embeddings[:n_start_vocab].shape,
      initializer=tf.constant_initializer(vocab.e_word.embeddings[:n_start_vocab]), trainable=True)
    # e_words_emb = tf.constant(vocab.e_word.embeddings[n_start_vocab:],
    #                           dtype=tf.float32)
    # j_words_emb = tf.constant(vocab.j_word.embeddings[n_start_vocab:], 
    #                           dtype=tf.float32)
    e_words_emb = self.initialize_embeddings(
      'EnWords', vocab.e_word.embeddings[n_start_vocab:].shape,
      initializer=tf.constant_initializer(vocab.e_word.embeddings[n_start_vocab:]),
      trainable=config.train_embedding
    )
    j_words_emb = self.initialize_embeddings(
      'JPWords', vocab.j_word.embeddings[n_start_vocab:].shape,
      initializer=tf.constant_initializer(vocab.j_word.embeddings[n_start_vocab:]),
      trainable=config.train_embedding
    )

    self.embeddings.e_word = tf.concat([special_tokens_emb, e_words_emb], axis=0)
    self.embeddings.j_word = tf.concat([special_tokens_emb, j_words_emb], axis=0)

  def setup_word_encoder(self, config, scope=None):
    word_repls = []
    with tf.variable_scope('Word') as scope:
      w_inputs = tf.nn.embedding_lookup(self.embeddings.e_word, self.e_inputs_w_ph)
      word_encoder = WordEncoder(self.keep_prob, shared_scope=scope)
      word_repls.append(word_encoder.encode(w_inputs))
    word_repls = tf.concat(word_repls, axis=-1) 
    return word_repls

  def setup_uttr_encoder(self, config, word_repls, scope=None):
    encoder_type = getattr(encoder, config.encoder.utterance.encoder_type)
    self.uttr_encoder = encoder_type(config.encoder.utterance, 
                                     self.keep_prob, 
                                     shared_scope=scope)
    return self.uttr_encoder.encode(word_repls, self.uttr_lengths)

  def setup_decoder_states(self, config, encoder_outputs, encoder_state, 
                           scope=None):
    attention_states = encoder_outputs
    decoder_state = encoder_state
    return decoder_state, decoder_state, attention_states

  def setup_projection(self, config, scope=None):
    return None

  def setup_decoder(self, config, train_decoder_state, test_decoder_state,
                    embeddings,
                    encoder_input_lengths=None,
                    attention_states=None, 
                    projection_layer=None,
                    scope=None):
    batch_size = self.batch_size
    decoder_inputs_emb = tf.nn.embedding_lookup(embeddings, self.decoder_inputs)
    # TODO: 多言語対応にする時はbias, trainableをfalseにしてembeddingをconstantにしたい
    decoder_cell = setup_cell(config.decoder.cell_type, 
                              shape(train_decoder_state, -1), 
                              config.decoder.num_layers,
                              keep_prob=self.keep_prob)
    if projection_layer is None:
      with tf.variable_scope('projection') as scope:
        kernel = tf.transpose(embeddings, perm=[1, 0])
        projection_layer = SharedKernelDense(shape(embeddings, 0), 
                                             use_bias=False, trainable=False,
                                             shared_kernel=kernel)

    with tf.name_scope('Training'):
      train_decoder_cell = decoder_cell
      decoder_initial_state = train_decoder_state
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
      test_decoder_cell = decoder_cell
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
        test_decoder_state, multiplier=beam_width)

      decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        test_decoder_cell, embeddings, self.start_tokens, self.end_token, 
        decoder_initial_state,
        beam_width, output_layer=projection_layer,
        length_penalty_weight=config.length_penalty_weight)
      test_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder, impute_finished=False,
        maximum_iterations=config.utterance_max_len, scope=scope)
      predictions = test_decoder_outputs.predicted_ids
      #self.beam_scores = test_decoder_outputs.beam_search_decoder_output.scores
      # memo: 出力結果はbeam_scoresの低い順にならんでいて (負の値を取る)，概ねそれはちゃんと正確さと一致してそう？

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
      #self.d_outputs_ph: np.array(batch.responses),
      self.is_training: is_training,
      #self.speaker_changes_ph: np.array(batch.speaker_changes)
    }
    assert not np.any(np.isnan(batch.texts))
    feed_dict[self.e_inputs_w_ph] = np.array(batch.texts)

    #if self.c_vocab:
    #  feed_dict[self.e_inputs_c_ph] = np.array(batch.c_contexts)

    return feed_dict

  @timewatch()
  def train(self, data, do_update=True):
    self.debug = [self.uttr_lengths, self.decoder_inputs, self.targets, self.target_length, self.target_weights]

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
        sys.stderr.write('Got Nan loss.\n')
        for x in feed_dict:
          print x
          print feed_dict[x]
        print set([np.count_nonzero(x) for x in feed_dict[self.e_inputs_w_ph]])
        exit(1)

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
    e_predictions = []
    j_predictions = []
    num_steps = 0
    epoch_time = 0.0
    for i, batch in enumerate(data):
      feed_dict = self.get_input_feed(batch, False)
      # for x,resx in zip(self.debug, self.sess.run(self.debug, feed_dict)):
      #    print x
      #    print resx.shape
      # exit(1)

      t = time.time()
      inputs.append(batch.texts)
      batch_predictions = self.sess.run(self.e_predictions, feed_dict)
      batch_predictions = np.transpose(batch_predictions, (0, 2, 1)) 
      e_predictions.append(batch_predictions)
      batch_predictions = self.sess.run(self.j_predictions, feed_dict)
      batch_predictions = np.transpose(batch_predictions, (0, 2, 1)) 
      j_predictions.append(batch_predictions)


      epoch_time += time.time() - t
      num_steps += 1
    inputs = flatten(inputs)
    e_predictions = flatten(e_predictions)
    j_predictions = flatten(j_predictions)
    inputs = [self.vocab.e_word.id2sent(u, join=True) for u in inputs]
    outputs = inputs
    e_predictions = [[self.vocab.e_word.id2sent(r, join=True) for r in p] for p in e_predictions]
    j_predictions = [[self.vocab.j_word.id2sent(r, join=True) for r in p] for p in j_predictions]
    return (inputs, outputs, e_predictions, j_predictions), epoch_time


#FinalBeamDecoderOutput(predicted_ids=<tf.Tensor 'Decoder/Test/projection/transpose:0' shape=(?, ?, 5) dtype=int32>, beam_search_decoder_output=BeamSearchDecoderOutput(scores=<tf.Tensor 'Decoder/Test/projection/transpose_1:0' shape=(?, ?, 5) dtype=float32>, predicted_ids=<tf.Tensor 'Decoder/Test/projection/transpose_2:0' shape=(?, ?, 5) dtype=int32>, parent_ids=<tf.Tensor 'Decoder/Test/projection/transpose_3:0' shape=(?, ?, 5) dtype=int32>))
