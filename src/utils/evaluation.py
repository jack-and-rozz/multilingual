# coding:utf-8
import sys, os, collections, re, copy
from utils.common import transpose, flatten, get_ngram

#golds = [['30', 'million'], ['60', 'million'], ['$'], ['-']]],   
#predictions = [(['30'], ['30'],  ['$'], ['-']), ()]


def exact_match(golds, predictions):
  assert len(golds) == len(predictions)
  n_examples = len(golds)
  golds = transpose(golds)
  predictions = transpose(predictions)
  res = []
  for golds_per_col, preds_per_col in zip(golds, predictions):
    n_match = len([True for g, p in zip(golds_per_col, preds_per_col) if g.strip() == p.strip()])
    res.append(n_match)
  return [float(n_match)/n_examples for n_match in res]

def precision_recall(golds, predictions, N=1):
  assert len(golds) == len(predictions)
  golds = transpose(golds)
  predictions = transpose(predictions)
  res = []

  precisions, recalls = [], []
  for golds_per_col, preds_per_col in zip(golds, predictions):
    TP = []
    FP = []
    FN = []
    for g, p in zip(golds_per_col, preds_per_col):
      tp = []
      g_ngrams = flatten(get_ngram(g, 1, N))
      p_ngrams = flatten(get_ngram(p, 1, N))
      for pn in copy.deepcopy(p_ngrams):
        if pn in g_ngrams:
          tp.append(pn)
          g_ngrams.pop(g_ngrams.index(pn))
          p_ngrams.pop(p_ngrams.index(pn))
      fn = g_ngrams
      fp = p_ngrams
      TP.append(tp)
      FP.append(fp)
      FN.append(fn)
    TP = len(flatten(TP))
    FP = len(flatten(FP))
    FN = len(flatten(FN))
    precisions.append(1.0*TP/(TP+FP))
    recalls.append(1.0*TP/(TP+FN))
  return precisions, recalls


if __name__ == '__main__':
  import random
  N = 5
  golds = [[str(random.randint(0, 10)) for i in range(N)]]
  preds = golds
  print exact_match(golds, preds)
  print precision_recall(golds, preds)
