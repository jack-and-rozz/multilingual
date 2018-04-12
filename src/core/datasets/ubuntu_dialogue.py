# coding:utf-8
import tensorflow as tf
import sys, re, random, itertools, os
import pandas as pd
from collections import OrderedDict, Counter
from core.vocabularies import _BOS, BOS_ID, _PAD, PAD_ID, _NUM,  WordVocabulary, CharVocabulary, BooleanVocab
from utils import common
from core.datasets.base import DatasetBase, PackedDatasetBase, _EOU, _EOT, _URL, _FILEPATH, w_dialogue_padding, c_dialogue_padding

class _UbuntuDialogueDataset(DatasetBase):
  def __init__(self, info, w_vocab, c_vocab, context_max_len=0):
    self.context_max_len = context_max_len
    self.sc_vocab = BooleanVocab
    DatasetBase.__init__(self, info, w_vocab, c_vocab)

  def preprocess(self, df):
    col_response = 'Utterance' if 'Utterance' in df else 'Ground Truth Utterance'
    data = []
    for x in zip(df['Context'], df[col_response]):
      d = self.preprocess_dialogue(x, context_max_len=self.context_max_len)
      if d:
        data.append(d)
    data = common.flatten(data)
    dialogues, speaker_changes = list(zip(*data))
    contexts, responses, speaker_changes = zip(*[(d[:-1], d[-1], sc[:-1]) for d, sc in zip(dialogues, speaker_changes) if sc[-1] == True])
    return contexts, responses, speaker_changes

  @classmethod
  def preprocess_dialogue(self, line, context_max_len=0, split_turn=True):
    speaker_changes = []
    context, response = line
    dialogue = _EOT.join([context, response])
    dialogue = [self.preprocess_turn(x.strip(), split_turn) 
                for x in dialogue.split(_EOT) if x.strip()]
    speaker_change = [[True if i == 0 else False for i in xrange(len(d))] for d in dialogue] # Set 1 when a speaker start his/her turn, otherwise 0.

    dialogue = common.flatten(dialogue)
    speaker_change = common.flatten(speaker_change)

    # The maximum length of a dialogue is context_max_len + 1 (response).
    dialogue_max_len = context_max_len + 1 if context_max_len else 0
    if not dialogue_max_len or len(dialogue) < dialogue_max_len:
      return [(dialogue, speaker_change)]
    else: # Slice the dialogue.
      res = common.flatten([[(dialogue[i:i+dlen], speaker_change[i:i+dlen]) for i in xrange(len(dialogue)+1-dlen)] for dlen in range(2, dialogue_max_len+1)])
      return res

  @classmethod
  def preprocess_turn(self, turn, split_turn):
    if split_turn:
      turn = [self.preprocess_utterance(uttr) for uttr in turn.split(_EOU) if uttr.strip()]
    else:
      turn = [self.preprocess_utterance(turn)]
    return turn

  @classmethod
  def preprocess_utterance(self, uttr):
    def _replace_pattern(uttr, before, after):
      m = re.search(before, uttr)
      if m:
        uttr = uttr.replace(m.group(0), after)
      return uttr

    patterns = [
      ('https?\s*:\s*/\S+/\S*', _URL),
      ('[~.]?/\S*', _FILEPATH),
    ]
    for before, after in patterns:
      uttr = _replace_pattern(uttr, before, after)

    replacements = [
      ("’", "'"),
      ("'", " ' "),
      (".", " . "),
      ("-", " - "),
      ("*", " * "),
      (_NUM, _NUM + ' ')
    ]
    for x, y in replacements:
      uttr = uttr.replace(x, y)

    return uttr.strip()

  @common.timewatch()
  def load_data(self, context_max_len=0):
    self.load = True
    sys.stderr.write('Loading dataset from %s ...\n' % (self.path))
    df = pd.read_csv(self.path, nrows=self.max_lines)

    sys.stderr.write('Preprocessing ...\n')
    if 'Label' in df:
      df = df[df['Label'] == 1]
    contexts, responses, speaker_changes = self.preprocess(df)

    if not self.wbase and not self.cbase:
      raise ValueError('Either \'wbase\' or \'cbase\' must be True.')

    self.speaker_changes = [self.sc_vocab.sent2id(sc) for sc in speaker_changes]

    # Separate contexts and responses into words (or chars), and convert them into their IDs.
    self.original = common.dotDict({})
    self.symbolized = common.dotDict({})

    if self.wbase:
      self.original.w_contexts = [[self.w_vocab.tokenizer(u) for u in context] 
                                  for context in contexts]
      self.symbolized.w_contexts = [[self.w_vocab.sent2id(u) for u in context] 
                                    for context in self.original.w_contexts]
    else:
      self.original.w_contexts = [None for context in contexts] 
      self.symbolized.w_contexts = [None for context in contexts] 

    if self.cbase:
      self.original.c_contexts = [[self.c_vocab.tokenizer(u) for u in context] 
                                  for context in contexts]

      self.symbolized.c_contexts = [[self.c_vocab.sent2id(u) for u in context] 
                                    for context in self.original.c_contexts]
    else:
      self.original.c_contexts = [None for context in contexts]
      self.symbolized.c_contexts = [None for context in contexts]
    self.original.responses = [self.w_vocab.tokenizer(r) for r in responses]
    self.symbolized.responses = [self.w_vocab.sent2id(r) for r in responses]

  def get_batch(self, batch_size, word_max_len=0,
                utterance_max_len=0, shuffle=False):
    if not self.load:
      self.load_data(context_max_len=context_max_len) # lazy loading.

    w_contexts = self.symbolized.w_contexts
    c_contexts = self.symbolized.c_contexts if self.cbase else [None for _ in xrange(len(w_contexts))]
    responses = self.symbolized.responses
    speaker_changes = self.speaker_changes

    data = [tuple(x) for x in zip(w_contexts, c_contexts, responses, speaker_changes, self.original.w_contexts, self.original.responses)]
    if shuffle: # For training.
      random.shuffle(data)
    for i, b in itertools.groupby(enumerate(data), 
                                  lambda x: x[0] // (batch_size)):
      batch = [x[1] for x in b]
      # Set the maximum length in the batch as *_max_len if it is not given.
      w_contexts, c_contexts, responses, speaker_changes, ori_w_contexts, ori_responses = zip(*batch)

      if self.wbase:
        w_contexts = w_dialogue_padding(w_contexts, self.context_max_len, 
                                        utterance_max_len)
      if self.cbase:
        c_contexts = c_dialogue_padding(c_contexts, self.context_max_len,
                                        utterance_max_len, word_max_len)

      _utterance_max_len = max([len(u) for u in responses]) 
      if not utterance_max_len or _utterance_max_len < utterance_max_len:
        utterance_max_len = _utterance_max_len

      responses = tf.keras.preprocessing.sequence.pad_sequences(
        responses, maxlen=utterance_max_len, 
        padding='post', truncating='post', value=PAD_ID)
      speaker_changes = tf.keras.preprocessing.sequence.pad_sequences(
        speaker_changes, maxlen=self.context_max_len,
        padding='post', truncating='post', value=PAD_ID)

      yield common.dotDict({
        'ori_w_contexts': ori_w_contexts,
        'ori_responses': ori_responses,
        'w_contexts': w_contexts,
        'c_contexts': c_contexts,
        'responses': responses,
        'speaker_changes': speaker_changes
      })

class UbuntuDialogueDataset(PackedDatasetBase):
  dataset_type = _UbuntuDialogueDataset
  @classmethod
  def get_words(self, train_data_path):
    dataset_type = self.dataset_type
    data = pd.read_csv(train_data_path)
    data = data[data['Label'] == 1]
    contexts, _ = list(zip(*[dataset_type.preprocess_dialogue(x) for x in data['Context']]))
    contexts = common.flatten(contexts)

    responses = [dataset_type.preprocess_turn(x) for x in data['Utterance']]
    responses = common.flatten(responses)
    texts = contexts + responses
    words = common.flatten([l.split() for l in texts])
    if type(texts[0]) == str:
      words = [word.decode('utf-8') for word in words] # List of unicode.
    return words

