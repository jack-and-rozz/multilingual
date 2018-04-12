import pyhocon
import numpy as np
import multiprocessing as mp
import time, random, os, commands, collections, re, sys
from datetime import datetime
from itertools import chain
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO, WARNING, ERROR, CRITICAL
try:
  import cPickle as pickle
except:
  import pickle

def transpose(l):
  return list(zip(*l))

def quote(str_):
  return "\"" + str_ + "\""

def timestamp():
  return datetime.now()

def separate_path_and_filename(file_path):
    pattern = '^(.+)/(.+)$'
    m = re.match(pattern, file_path)
    if m:
      path, filename = m.group(1), m.group(2) 
    else:
      path, filename = None , file_path
    return path, filename

def random_string(length, seq='0123456789abcdefghijklmnopqrstuvwxyz'):
    sr = random.SystemRandom()
    return ''.join([sr.choice(seq) for i in xrange(length)])

def restore_to_tmpfile(sentences, tmp_dir='/tmp'):
  """
  sentences: List of str.
  """
  tmp_filename = random_string(5)
  tmp_filepath = os.path.join(tmp_dir, tmp_filename)
  with open(tmp_filepath, 'w') as f:
    for line in sentences:
      if type(line) == list:
        line = ' '.join(line)
      f.write(line + '\n')
  return tmp_filepath

def get_pos(sents, output_path=None):
  suffix = '.pos'
  output_path = output_path + suffix
  sys.stderr.write(output_path+'\n')
  if output_path and os.path.exists(output_path):
    sys.stderr.write("Reading the POS file...\n")
    pos_tags = [[x for x in l.split('\n') if x] for l in commands.getoutput('cut -f2 %s' % (output_path)).split('\n\n')]
  else:
    sys.stderr.write("Runnning POS Tagger...\n")
    tmp_filepath = restore_to_tmpfile(sents, tmp_dir='/tmp')
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               'scripts/stanford-postagger.sh')
    cmd = "%s %s 4g" % (script_path, tmp_filepath) 
    os.system(cmd)
    pos_tags = [[x for x in l.split('\n') if x] for l in commands.getoutput('cut -f2 %s' % (tmp_filepath + suffix)).split('\n\n')]
    if output_path is not None:
      sys.stderr.write("Restoring to \'%s\' ...  \n" % output_path)
      os.system('cp %s %s' % (tmp_filepath + suffix, output_path))
    os.system('rm %s' % (tmp_filepath))
    os.system('rm %s%s' % (tmp_filepath, suffix))
  try:
    assert len(pos_tags) == len(sents)
  except:
    print  ("The length of POS and sources don't match each other. (%d, %d)"  % (len(pos_tags), len(sents)))
    if output_path:
      os.system('rm %s' % (output_path))
    exit(1)
  return pos_tags

def str2tuple(v):
  type_f = lambda x: int(x) if x.isdigit() else str(x)
  if type(v) in [list, tuple]:
    res = (type_f(x) for x in v if x != ',')
  else:
    m = re.match('\((.+)\)', v)
    if m:
      v = m.group(1)

    res = (re.sub('[\'\"]', '', x).strip() for x in v.split(','))
    res = tuple([type_f(x) for x in res if x])
  return tuple(list(res))

def str2bool(v):
  if type(v) == bool:
    return v
  return v.lower() in ("yes", "true", "t", "1")

def flatten(l, depth=1):
  # if depth is None:
  #   l = [flatten(x, depth) for x in l] if type(l[0]) in [list, tuple] else l
  # else:
  l = flatten(l, depth-1) if depth > 1 else l 
  return list(chain.from_iterable(l))


def logManager(logger_name='main', 
              handler=StreamHandler(),
              log_format = "[%(levelname)s] %(asctime)s - %(message)s",
              level=DEBUG):
    formatter = Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    logger = getLogger(logger_name)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def timewatch(logger=None):
    if logger is None:
      logger = logManager(logger_name='utils')
    def _timewatch(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.info("%s: %f sec" % (func.__name__ , end - start))
            return result
        return wrapper
    return _timewatch

def print_colored(sentence, idx_colored, color='red'):
  assert isinstance(sentence, list)
  res = []
  for i,s in enumerate(sentence):
    if i in idx_colored:
      res.append(colored(s, color))
    else:
      res.append(s)
  print (" ".join(res))

def colored(str_, color):
  '''
  Args: colors: a str or list of it.
  '''
  RESET = "\033[0m"
  ctable = {
    'black': "\033[30m",
    'red': "\033[31m",
    'green': "\033[32m",
    'yellow': "\033[33m",
    'blue': "\033[34m",
    'purple': "\033[35m",
    'underline': '\033[4m',
    'link': "\033[31m" + '\033[4m',
    'bold': '\033[30m' + "\033[1m",
  }
  if type(color) == str:
    res = ctable[color] + str_ + RESET
  elif type(color) == tuple or type(color) == list:
    res = "".join([ctable[c] for c in color]) + str_ + RESET
  return res 


class dotDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

class recDotDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__
  def __init__(self, _dict):
    for k in _dict:
      if isinstance(_dict[k], dict):
        _dict[k] = recDotDict(_dict[k])
      # if isinstance(_dict[k], list):
      #    for i,x in enumerate(_dict[k]):
      #      print i, x, isinstance(x, dict)
      #      if isinstance(x, dict):
      #        dict[k][i] = dotDict(x)
    super(recDotDict, self).__init__(_dict)

class OrderedDefaultDict(collections.OrderedDict, collections.defaultdict):
  def __init__(self, default_factory=None, *args, **kwargs):
    #in python3 you can omit the args to super
    super(OrderedDefaultDict, self).__init__(*args, **kwargs)
    self.default_factory = default_factory


def multi_process(func, *args):
  '''
  Args:
    - func : a function to be executed.
    - args : a list of list of args that a worker needs.
             [[id1, name1, ...], [id2, name2, ...]]
  '''
  # A wrapper to make a function put its response to a queue.
  def wrapper(_func, idx, q):
    def _wrapper(*args, **kwargs):
      res = func(*args, **kwargs)
      return q.put((idx, res))
    return _wrapper

  workers = []
  # mp.Queue() seems to have a bug..?
  # (stackoverflow.com/questions/13649625/multiprocessing-in-python-blocked)
  q = mp.Manager().Queue()

  # kwargs are not supported... (todo)
  for i, a in enumerate(zip(*args)):
    worker = mp.Process(target=wrapper(func, i, q), args=a)
    workers.append(worker)
    worker.daemon = True  # make interrupting the process with ctrl+c easier
    worker.start()

  for worker in workers:
    worker.join()
  results = []
  while not q.empty():
    res = q.get()
    results.append(res)

  return [res for i, res in sorted(results, key=lambda x: x[0])]


def get_ngram_match(sent, ngram):
  if type(sent) == str:
    sent = sent.split(' ')
  n = len(ngram)
  indices = [(i, i+n-1) for i in xrange(len(sent)-n+1) if tuple(sent[i:i+n]) == ngram]
  return indices

def get_ngram(s, min_n, max_n, vocab_condition=lambda x: True):
  if type(s) == str:
    s = s.split(' ')
  ngrams = [[tuple(s[i:i+n]) for i in xrange(len(s)-n+1) if vocab_condition(s[i:i+n])] for n in xrange(min_n, max_n+1)]
  return ngrams

def no_overlaps(existing_spans, new_span):
  # Input : list of tuple [(int, int), ....], a tuple
  ns = new_span
  check = [True if es[1] < ns[0] or ns[1] < es[0] else False for es in existing_spans]
  return True if not False in check else False


def unzip(l):
    #*map(list, zip(sents, sents_pos))
  return map(list, zip(*l))


def ask_yn(message, func, *args, **kwargs):
  while True:
    sys.stderr.write(message + '\n')
    x = raw_input().lower()
    if x == 'y' or x == '\n' or x == '':
      func(*args, **kwargs)
      break
    elif x == 'n':
      sys.stderr.write("Operation was aborted.\n")
      exit(1)
    else:
      sys.stderr.write("Type 'y' or 'n'.\n")


def get_config(filename):
  return pyhocon.ConfigFactory.parse_file(filename)

def print_config(config):
  print (pyhocon.HOCONConverter.convert(config, "hocon"))
