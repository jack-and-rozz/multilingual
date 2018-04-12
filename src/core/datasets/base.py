#coding: utf-8
import tensorflow as tf
import sys, re, random, itertools, os
import pandas as pd
from collections import OrderedDict, Counter
from core.vocabularies import _BOS, BOS_ID, _PAD, PAD_ID, _NUM,  WordVocabulary, CharVocabulary
from utils import evaluation, tf_utils, common

max_cnn_width=5 # TODO 
_EOU = '__eou__'
_EOT = '__eot__'
_URL = '__URL__'
_FILEPATH = '__FILEPATH__'

def w_dialogue_padding(w_contexts, context_max_len, utterance_max_len):
  # Get maximum length of contexts and utterances.
  _context_max_len = max([len(d) for d in w_contexts])
  context_max_len = context_max_len if context_max_len else _context_max_len
  
  _utterance_max_len = max([max([len(u) for u in d]) for d in w_contexts]) 
  if not utterance_max_len or _utterance_max_len < utterance_max_len:
    utterance_max_len = _utterance_max_len

  # TODO: the sequences encoded by CNN must be longer than the filter size.
  utterance_max_len = max(max_cnn_width, utterance_max_len)

  # Fill empty utterances.
  w_contexts = [[d[i] if i < len(d) else [] for i in xrange(context_max_len)] for d in w_contexts]
  w_contexts = [tf.keras.preprocessing.sequence.pad_sequences(
    d, maxlen=utterance_max_len, 
    padding='post', truncating='post', value=PAD_ID) for d in w_contexts]
  return w_contexts
  
def c_dialogue_padding(c_contexts, context_max_len, utterance_max_len, 
                       word_max_len):
  # Get maximum length of contexts, utterances, and words.
  _context_max_len = max([len(d) for d in c_contexts])
  context_max_len = context_max_len if context_max_len else _context_max_len

  _utterance_max_len = max([max([len(u) for u in d]) for d in c_contexts]) 
  if not utterance_max_len or _utterance_max_len < utterance_max_len:
    utterance_max_len = _utterance_max_len
  _word_max_len = max([max([max([len(w) for w in u]) for u in d]) for d in c_contexts])
  # TODO: the sequences encoded by CNN must be longer than the filter size.
  utterance_max_len = max(max_cnn_width, utterance_max_len)
  word_max_len = max(max_cnn_width, word_max_len)

  if not word_max_len or _word_max_len < word_max_len:
    word_max_len = _word_max_len

  # Fill empty utterances.
  c_contexts = [[d[i] if i < len(d) else [] for i in xrange(context_max_len)] for d in c_contexts]
  c_contexts = [[[u[i] if i < len(u) else [] for i in xrange(utterance_max_len)] for u in d] for d in c_contexts]

  c_contexts = [[tf.keras.preprocessing.sequence.pad_sequences(
    u, maxlen=word_max_len, padding='post', truncating='post',
    value=PAD_ID) for u in d] for d in c_contexts]
  return c_contexts



class DatasetBase(object):
  def __init__(self, info, w_vocab, c_vocab):

    self.path = info.path
    self.max_lines = info.max_lines if info.max_lines else None
    self.w_vocab = w_vocab
    self.c_vocab = c_vocab
    self.wbase = w_vocab is not None
    self.cbase = c_vocab is not None
    self.load = False


###################################################
#    Classes for dataset pair (train, valid, test)
###################################################

class PackedDatasetBase(object):
  '''
  The class contains train, valid, test dataset.
  Each dataset class has different types of .
  args:
     dataset_type: A string. It is the name of dataset class defined in config.
     pathes: A list of string. ([train_path, valid_path, test_path])
  kwargs:
     num_train_data: The upperbound of the number of training examples. If 0, all of the data will be used.
     no_train: whether to omit to load training data to save time. (in testing)
  '''
  dataset_type = None
  @common.timewatch()
  def __init__(self, info, *args, **kwargs):
    if not self.dataset_type:
      raise ValueError('The derivative of PackedDatasetBase must have class variable \'\dataset_type\'.')
    dataset_type = self.dataset_type
    #dataset_type = getattr(core.datasets, '_' + self.__class__.__name__)
    self.train = dataset_type(info.train, *args, **kwargs) 
    self.valid = dataset_type(info.valid, *args, **kwargs)
    self.test = dataset_type(info.test, *args, **kwargs)

  @classmethod
  def create_vocab_from_data(self, config):
    train_data_path = config.dataset_info.train.path
    w_vocab_size = config.w_vocab_size
    c_vocab_size = config.c_vocab_size
    lowercase = config.lowercase

    w_vocab_path = train_data_path + '.Wvocab' + str(w_vocab_size)
    if lowercase:
      w_vocab_path += '.lower'
    c_vocab_path = train_data_path + '.Cvocab' + str(c_vocab_size)
    if not (os.path.exists(w_vocab_path) and os.path.exists(c_vocab_path)):
      words = self.get_words(train_data_path)
    else:
      words = ['-']

    w_vocab = WordVocabulary(w_vocab_path, words, vocab_size=w_vocab_size, lowercase=lowercase) if config.wbase else None
    c_vocab = CharVocabulary(c_vocab_path, words, vocab_size=c_vocab_size, lowercase=False, normalize_digits=False) if config.cbase else None
    return w_vocab, c_vocab

  @classmethod
  def get_words(self, train_data_path):
    raise NotImplementedError


