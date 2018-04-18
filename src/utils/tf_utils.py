
#coding: utf-8
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

def shape(x, dim):
  return x.get_shape()[dim].value or tf.shape(x)[dim]

def flatten(tensor, target_rank):
  # Convert the rank of a tensor (>= target_rank) to target_rank.
  # e.g. (tensor.get_shape() = [10, 5, 10, 100], target_rank=3) -> [50, 10, 100]

  rank = len(tensor.get_shape())
  prev_shape = [shape(tensor, i) for i in xrange(rank)]
  assert rank >= target_rank
  with tf.name_scope('flatten'):
    if rank > target_rank:
      flattened_dims = [shape(tensor, i) for i in xrange(rank-target_rank+1)]
      #unchanged_dims = [shape(tensor, i) for i in xrange(target_rank, rank)]
      unchanged_dims = [shape(tensor, rank-target_rank+1+i) for i in xrange(target_rank-1)]
      flattened_shape = [np.prod(flattened_dims)] + unchanged_dims
      flattened_tensor = tf.reshape(tensor, flattened_shape)
    else:
      flattened_tensor = tensor
    return flattened_tensor, prev_shape

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def make_summary(value_dict):
  return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in value_dict.items()])

def linear(inputs, output_size, add_bias=True,
           activation=tf.nn.relu, scope=None):
  """
  Args:
    inputs : Rank 2 or 3 Tensor of shape [batch_size, (sequence_size,) hidden_size].
                 the sequence_size must be known.
    output_size : An integer.
  """
  if activation is None:
    activation = lambda x: x
  with tf.variable_scope(scope or "linear"):
    inputs_rank = len(inputs.get_shape().as_list())
    hidden_size = shape(inputs, -1)
    w = tf.get_variable('weights', [hidden_size, output_size])
    if add_bias:
      b = tf.get_variable('biases', [output_size])
    else:
      b = tf.constant(0, shape=[output_size], dtype=tf.float32)

    if inputs_rank == 3:
      batch_size = shape(inputs, 0)
      max_sentence_length = shape(inputs, 1)
      inputs = tf.reshape(inputs, [batch_size * max_sentence_length, hidden_size])
      outputs = activation(tf.nn.xw_plus_b(inputs, w, b))
      outputs = tf.reshape(outputs, [batch_size, max_sentence_length, output_size])
    elif inputs_rank == 2:
      outputs = activation(tf.nn.xw_plus_b(inputs, w, b))
    else:
      ValueError("linear with rank {} not supported".format(inputs_rank))

    #if out_keep_prob is not None and out_keep_prob < 1.0:
    return outputs

def cnn(inputs, filter_sizes=[3, 4, 5], num_filters=50, activation=tf.nn.relu):
  num_words = shape(inputs, 0)
  num_chars = shape(inputs, 1)
  input_size = shape(inputs, 2)
  outputs = []
  with tf.variable_scope('CNN'):
    for i, filter_size in enumerate(filter_sizes):
      with tf.variable_scope("conv_width{}".format(filter_size)):
        w = tf.get_variable("w", [filter_size, input_size, num_filters])
        b = tf.get_variable("b", [num_filters])
      conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID") # [num_words, num_chars - filter_size, num_filters]
      h = activation(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
      pooled = tf.reduce_max(h, 1) # [num_words, num_filters]
      outputs.append(pooled)
  return tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]
  #return tf.reshape(tf.concat(outputs, 1), [num_words, num_filters * len(filter_sizes)])

def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
  if len(inputs.get_shape()) > 2:
    current_inputs = tf.reshape(inputs, [-1, shape(inputs, -1)])
  else:
    current_inputs = inputs

  with tf.variable_scope('FFNN'):
    for i in xrange(num_hidden_layers):
      hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size])
      hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size])
      current_outputs = tf.nn.relu(tf.matmul(current_inputs, hidden_weights) + hidden_bias)

      if dropout is not None:
        current_outputs = tf.nn.dropout(current_outputs, dropout)
      current_inputs = current_outputs

    output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size], initializer=output_weights_initializer)
    output_bias = tf.get_variable("output_bias", [output_size])
    outputs = tf.matmul(current_inputs, output_weights) + output_bias

    if len(inputs.get_shape()) == 3:
      outputs = tf.reshape(outputs, [shape(inputs, 0), shape(inputs, 1), output_size])
    elif len(inputs.get_shape()) > 3:
      raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))
  return outputs

def projection(inputs, output_size, initializer=None):
  return ffnn(inputs, 0, -1, output_size, dropout=None, output_weights_initializer=initializer)


def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)


from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
class SharedKernelDense(tf.layers.Dense):
  def __init__(self, units, shared_kernel=None, **kwargs):
    super(SharedKernelDense, self).__init__(units, **kwargs)
    self.kernel = None
    self.shared_kernel = shared_kernel

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = base.InputSpec(min_ndim=2,
                                     axes={-1: input_shape[-1].value})
    if self.shared_kernel is None:
      self.kernel = self.add_variable('kernel',
                                      shape=[input_shape[-1].value, self.units],
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype=self.dtype,
                                      trainable=True)
    else:
      self.kernel = self.shared_kernel
    if self.use_bias:
      self.bias = self.add_variable('bias',
                                    shape=[self.units,],
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    dtype=self.dtype,
                                    trainable=True)
    else:
      self.bias = None
    self.built = True
