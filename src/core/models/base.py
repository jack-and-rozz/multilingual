import math
import tensorflow as tf
from core.extensions.pointer import pointer_decoder

def setup_cell(cell_type, size, num_layers, keep_prob=None):
  def _get_single_cell():
    cell = getattr(tf.contrib.rnn, cell_type)(size)
    if keep_prob is not None:
      cell = tf.contrib.rnn.DropoutWrapper(cell, 
                                           input_keep_prob=1.0,
                                           output_keep_prob=keep_prob)
    return cell

  if num_layers > 1:
    cell = tf.contrib.rnn.MultiRNNCell([_get_single_cell() for _ in range(num_layers)]) 
  else:
    cell = _get_single_cell()
  return cell

class ModelBase(object):
  def __init__(self, sess, config):
    self.sess = sess
    self.max_gradient_norm = config.max_gradient_norm
    with tf.variable_scope('GlobalVariables'):
      self.global_step = tf.get_variable(
        "global_step", trainable=False, shape=[],  dtype=tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32)) 

      self.epoch = tf.get_variable(
        "epoch", trainable=False, shape=[], dtype=tf.int32,
        initializer=tf.constant_initializer(0, dtype=tf.int32)) 

      self.high_score = tf.get_variable(
        "high_score", trainable=False, shape=[], dtype=tf.float32,
        initializer=tf.constant_initializer(0, dtype=tf.int32)) 

      self.learning_rate = tf.train.exponential_decay(
        config.learning_rate, self.global_step,
        config.decay_frequency, config.decay_rate, staircase=True)

    self.is_training = tf.placeholder(tf.bool, [], name='is_training')
    with tf.name_scope('keep_prob'):
      self.keep_prob = 1.0 - tf.to_float(self.is_training) * config.dropout_rate


  def get_updates(self, loss):
    params = tf.contrib.framework.get_trainable_variables()
    opt = tf.train.AdamOptimizer(self.learning_rate)
    gradients = [grad for grad, _ in opt.compute_gradients(loss)]
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
                                                    self.max_gradient_norm)
    grad_and_vars = [(g, v) for g, v in zip(clipped_gradients, params)]
    updates = opt.apply_gradients(
      grad_and_vars, global_step=self.global_step)
    return updates

  def add_epoch(self):
    self.sess.run(tf.assign(self.epoch, tf.add(self.epoch, tf.constant(1, dtype=tf.int32))))
  def update_highscore(self, score):
    self.sess.run(tf.assign(self.high_score, tf.constant(score, dtype=tf.float32)))

  def initialize_embeddings(self, name, emb_shape, initializer=None, 
                            trainable=True):
    if not initializer:
      initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
    embeddings = tf.get_variable(name, emb_shape, trainable=trainable,
                                 initializer=initializer)
    return embeddings
