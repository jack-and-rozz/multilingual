# coding: utf-8
import argparse, re, sys, os
import pandas as pd


def main(args):
  i = 0
  idx, sent, label = None, None, None
  res = []
  for l in open(args.file_path):
    m = re.match('^<(\d+)>', l)
    m2 = re.match('^Test input       :\t(.+)', l)
    m3 = re.match('^Human label      :\t(.+)', l)
    if m:
      idx = m.group(1)
    if m2: 
      sent = m2.group(1)
    if m3:
      label = [x.strip() for x in m3.group(1).split('|')]
    if idx and sent and label:
      res.append([idx, sent] + label)
      idx, sent, label = None, None, None
  indices, sents, lb, ub, currency, rate = list(zip(*res))
  df = pd.DataFrame({
    'index': indices,
    'sentence': sents,
    'LB':lb,
    'UB':ub,
    'currency':currency,
    'rate':rate
  }).ix[:, ['index', 'sentence', 'LB', 'UB', 'currency', 'rate']].set_index('index')

  print df.to_csv()
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file_path")
  args  = parser.parse_args()
  main(args)


