# coding:utf-8
import tensorflow as tf
import collections, os, time, re, sys, math
from tensorflow.python.platform import gfile
from orderedset import OrderedSet
from nltk.tokenize import word_tokenize
import numpy as np

import utils.common as common

_PAD = "_PAD"
_BOS = "_BOS"
_EOS = "_EOS"
_UNK = "_UNK"
_NUM = "_NUM"

ERROR_ID = -1
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
NUM_ID = 4 

_DIGIT_RE = re.compile(r"\d")
START_VOCAB = [_PAD, _UNK, _BOS, _EOS, _NUM]
UNDISPLAYED_TOKENS = [_PAD, _BOS, _EOS]

def separate_numbers(sent):
  '''
  Since for some reason nltk.tokenizer fails to separate numbers (e.g. 6.73you),
  manually separate them.
  Args:
     sent: a string.
  '''
  for m in re.findall("(\D*)(\d?[0-9\,\.]*\d)(\D*?)", sent):
  #for m in re.findall("(\D*)(\d?[0-9\,\.]*\d)(\D*)", sent):
    m = [x for x in m if x]
    sent = sent.replace(''.join(m), ' ' + ' '.join(m)+ ' ')
    sent = ' '.join(sent.split())
  return sent

def separate_symbols(sent):
  symbols = ['/', ':']
  for sym in symbols:
    sent = sent.replace(sym, " %s " % sym)
  return ' '.join(sent.split())

class WordTokenizer(object):
  def __init__(self, lowercase=False, normalize_digits=False,
               do_separate_numbers=False, do_separate_symbols=False):
    self.lowercase = lowercase
    self.normalize_digits = normalize_digits
    self.do_separate_numbers = do_separate_numbers
    self.do_separate_symbols = do_separate_symbols
    self.word_tokenize = lambda x : x.split() # Just split
    #self.word_tokenize = word_tokenize # Use nltk.word_tokenize

  def __call__(self, sent, normalize_digits=None, lowercase=None):
    sent = sent.replace('\n', '')
    if self.do_separate_numbers:
      sent = separate_numbers(sent)
    if self.do_separate_symbols:
      sent = separate_symbols(sent)
    normalize_digits = normalize_digits if normalize_digits is not None else self.normalize_digits
    lowercase = lowercase if lowercase is not None else self.lowercase
    if lowercase:
      sent = sent.lower()
    if normalize_digits:
      sent = re.sub(_DIGIT_RE, "0", sent) 
      for m in sorted(re.findall("0+", sent), key=lambda x: -len(x)):
        sent = sent.replace(m, _NUM)
    if type(sent) == str:
      sent = sent.decode('utf-8')
    return self.word_tokenize(sent)

class CharTokenizer(object):
  def __init__(self, lowercase=False, normalize_digits=False):
    self.lowercase = lowercase
    self.normalize_digits = normalize_digits
    self.w_tokenizer = WordTokenizer(lowercase=lowercase, 
                                     normalize_digits=normalize_digits)

  def word2chars(self, word):
    word = word.strip()
    if type(word) == str:
      word = word.decode('utf-8')
    word = [c for c in word]
    return word

  def __call__(self, sent, normalize_digits=None, lowercase=None):
    words = self.w_tokenizer(sent, lowercase=lowercase, 
                             normalize_digits=normalize_digits)

    return [self.word2chars(w) for w in words]

class VocabularyBase(object):
  def __init__(self):
    self.vocab = None
    self.rev_vocab = None
    self.embeddings = None
    self.name = None

  @property
  def vocab_size(self):
    return len(self.vocab)

  @property
  def emb_size(self):
    return len(self.embeddings[0]) if self.embeddings is not None else None

class WordVocabularyBase(VocabularyBase):
  def id2word(self, _id):
    if not type(_id) in [int, np.int32]:
      raise ValueError('ID must be an integer but %s' % str(type(_id)))
    elif _id < 0 or _id > len(self.rev_vocab):
      return None
      #raise ValueError('Token ID must be an integer between 0 and %d (ID=%d)' % (len(self.rev_vocab), _id))
    # elif _id in set([PAD_ID, EOS_ID, BOS_ID]):
    #   return None
    else:
      return self.rev_vocab[_id]

  def id2sent(self, ids, link_span=None, join=False, remove_special=True):
    '''
    ids: a list of word-ids.
    link_span : a tuple of the indices between the start and the end of a link.
    '''
    def _id2sent(ids, link_span):
      sent_tokens = [self.id2word(word_id) for word_id in ids if self.id2word(word_id) is not None]
      if link_span:
        for i in xrange(link_span[0], link_span[1]+1):
          sent_tokens[i] = common.colored(sent_tokens[i], 'link')
      if remove_special:
        sent_tokens = [w for w in sent_tokens 
                       if w not in UNDISPLAYED_TOKENS]
      if join:
        sent_tokens = " ".join(sent_tokens)
      return sent_tokens
    return _id2sent(ids, link_span)

  def word2id(self, word):
    # Args:
    #  - token: a string.
    # Res: an interger.
    return self.vocab.get(word, self.vocab.get(_UNK))

  def sent2id(self, sentence):
    if type(sentence) == list:
      res = [self.word2id(word) for word in sentence]
    elif type(sentence) == str:
      res = [self.word2id(word) for word in self.tokenizer(sentence)]
    elif type(sentence) == tf.Tensor and self.lookup_table:
      res = self.lookup_table.lookup(sentence)
    else:
      raise ValueError
    return res

class WordVocabulary(WordVocabularyBase):
  tokenizer_type = WordTokenizer
  @common.timewatch()
  def __init__(self, texts, vocab_path=None, start_vocab=START_VOCAB,
               vocab_size=0, lowercase=False, normalize_digits=True):
    """
    Args:
    - texts: List of tokens.
    """
    self.start_vocab = start_vocab
    WordVocabularyBase.__init__(self)
    self.tokenizer = self.tokenizer_type(lowercase=lowercase,
                                         normalize_digits=normalize_digits)
    self.vocab, self.rev_vocab, _ = self.init_vocab(
      texts, vocab_path, vocab_size=vocab_size)
    self.vocab_path = vocab_path

  def init_vocab(self, texts, vocab_path, vocab_size=0):
    if vocab_path and os.path.exists(vocab_path):
      sys.stderr.write('Loading word vocabulary from %s...\n' % vocab_path)
      vocab, rev_vocab = self.load_vocab(vocab_path)
    else:
      sys.stderr.write('Restoring word vocabulary to %s...\n' % vocab_path)
      vocab, rev_vocab = self.create_vocab(texts, vocab_path, 
                                           vocab_size=vocab_size)
    embeddings = None
    return vocab, rev_vocab, embeddings

  def load_vocab(self, vocab_path):
    rev_vocab = [l.replace('\n', '').split('\t')[0] for l in open(vocab_path)]
    vocab = collections.OrderedDict()
    for i,t in enumerate(rev_vocab):
      vocab[t] = i
    return vocab, rev_vocab
    
  def create_vocab(self, texts, vocab_path, vocab_size=0):
    '''
    Args:
     - vocab_path: The path to which the vocabulary will be restored.
     - texts: List of words.
    '''
    start_vocab = self.start_vocab
    rev_vocab, freq = zip(*collections.Counter(texts).most_common())
    rev_vocab = common.flatten([self.tokenizer(w) for w in rev_vocab])
    if type(rev_vocab[0]) == list:
      rev_vocab = common.flatten(rev_vocab)
    rev_vocab = OrderedSet(start_vocab + rev_vocab)
    if vocab_size:
      rev_vocab = OrderedSet([w for i, w in enumerate(rev_vocab) if i < vocab_size])
    freq = [0 for _ in start_vocab] + list(freq)
    freq = freq[:len(rev_vocab)]
    vocab = collections.OrderedDict()
    for i,t in enumerate(rev_vocab):
      vocab[t] = i

    # Restore vocabulary.
    if vocab_path is not None:
      with open(vocab_path, 'w') as f:
        for k, v in zip(rev_vocab, freq):
          if type(k) == unicode:
            k = k.encode('utf-8')
          f.write('%s\t%d\n' % (k,v))
    return vocab, rev_vocab

class CharVocabulary(WordVocabulary):
  tokenizer_type = CharTokenizer
  def id2word(self, _ids):
    chars = [WordVocabulary.id2word(self, _id) for _id in _ids]
    return ''.join([c for c in chars if not c in UNDISPLAYED_TOKENS])

  def word2id(self, chars):
    '''
    chars: A word represented as a list of characters.
    '''
    return [WordVocabulary.word2id(self, c) for c in chars]

  def create_vocab(self, vocab_path, texts, vocab_size=0):
    texts = common.flatten([self.tokenizer.word2chars(word) for word in texts])
    return WordVocabulary.create_vocab(self, vocab_path, texts, vocab_size=vocab_size)


class WordVocabularyWithEmbedding(WordVocabulary):
  def __init__(self, vocab_path, skip_first=True,
               vocab_size=0, lowercase=False, normalize_digits=True):
    WordVocabularyBase.__init__(self)
    self.tokenizer = self.tokenizer_type(lowercase=lowercase,
                                         normalize_digits=normalize_digits)
    self.vocab, self.rev_vocab, self.embeddings = self.init_vocab(
      vocab_path, skip_first, vocab_size=vocab_size)
    self.vocab_path = vocab_path

  def init_vocab(self, vocab_path, skip_first, 
                 vocab_size=0, start_vocab=START_VOCAB):
    sys.stderr.write('Loading word vocabulary from %s...\n' % vocab_path)
    self.start_vocab = start_vocab
    vocab, rev_vocab, embeddings = self.load_vocab(vocab_path, vocab_size=vocab_size, skip_first=skip_first)
    _vocab_path = vocab_path + '.Wvocab%d' % vocab_size
    if self.tokenizer.lowercase:
      _vocab_path += '.lower'
    if self.tokenizer.normalize_digits:
      _vocab_path += '.normD'
    assert len(set([len(vocab), len(rev_vocab), len(embeddings)])) == 1
    if not os.path.exists(_vocab_path):
      sys.stderr.write('Restoring word vocabulary to %s...\n' % _vocab_path)
      self.save_vocab(rev_vocab, embeddings, _vocab_path)
    return vocab, rev_vocab, embeddings

  def save_vocab(self, rev_vocab, embeddings, vocab_path):
    with open(vocab_path, 'w') as f:
      for v, e in zip(rev_vocab, embeddings):
        line = "%s %s\n" % (v, ' '.join([str(x) for x in e]))
        line = line.encode('utf-8')
        f.write(line)

  def load_vocab(self, embedding_path, skip_first=True, vocab_size=0):
    '''
    Load pretrained vocabularies and embeddings.
    '''
    sys.stderr.write("Loading word embeddings from {}...\n".format(embedding_path))
    word_and_embedding = []
    with open(embedding_path) as f:
      for i, line in enumerate(f):
        if skip_first and i == 0:
          continue
        if vocab_size and len(word_and_embedding) >= vocab_size:
          break
        line = line.split()
        word, embedding = line[0], line[1:]
        word = self.tokenizer(word)
        if len(word) != 1:
          continue
        else:
          word = word[0]
        embedding = [float(s) for s in embedding]
        embedding_size = len(embedding)
        word_and_embedding.append((word, embedding))

    embedding_dict = common.OrderedDefaultDict(
      default_factory=lambda:np.random.uniform(-math.sqrt(3), math.sqrt(3),
                                               size=embedding_size))
    for s in self.start_vocab:
      embedding_dict[s] = embedding_dict[s]
    for word, embedding in word_and_embedding:
      if word not in embedding_dict:
        embedding_dict[word] = embedding
    # print collections.Counter([type(v) for k, v in embedding_dict.items()])
    # print embedding_dict.values().shape
    # for s in START_VOCAB:
    #   print embedding_dict
    # exit(1)
    rev_vocab = embedding_dict.keys()
    embeddings = np.array(embedding_dict.values())
    vocab = collections.OrderedDict()
    for i,t in enumerate(rev_vocab):
      vocab[t] = i
    sys.stderr.write("Done loading word embeddings.\n")
    
    return vocab, rev_vocab, embeddings



class BooleanVocabulary(WordVocabularyBase):
  def __init__(self, start_vocab=[_PAD, _UNK]):
    self.start_vocab = start_vocab
    self.rev_vocab = start_vocab + [True, False]
    self.vocab = collections.OrderedDict([(v, i) for i, v in enumerate(self.rev_vocab)]) 
  def id2sent(self, ids, link_span=None, join=False, remove_special=True):
    return super(BooleanVocabulary, self).id2sent(ids, link_span=link_span, join=False, remove_special=remove_special)

BooleanVocab = BooleanVocabulary()

