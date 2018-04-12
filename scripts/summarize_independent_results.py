# coding: utf-8
import argparse, re, sys, os
import pandas as pd
import commands

def main(args):
  golds = []
  preds = []
  stats = []
  output_path = args.root_path + '/tests'

  for model_path in commands.getoutput('ls -d %s/*' % args.root_path).split('\n'):
    if model_path == output_path:
      continue
    file_path = model_path + '/tests/' + args.result_file
    indices, sents, token_sents, col_golds, col_preds, col_stats = read(file_path)
    stats.append(col_stats)
    golds.append(col_golds)
    preds.append(col_preds)
  golds = zip(*golds)
  preds = zip(*preds)
  stats.insert(0, ['Metrix', 'EM accuracy', 'Precision', 'Recall'])
  stats = zip(*stats)
  header = stats[0]
  stats = zip(*stats[1:])
  df = pd.DataFrame({k:v for k,v in zip(header, stats)}).ix[:, header].set_index('Metrix')

  if not os.path.exists(output_path):
    os.makedirs(output_path)
  with open(os.path.join(output_path, args.result_file), 'w') as f:
    sys.stdout = f
    for idx, sent, token_sent, gold, pred in zip(indices, sents, token_sents, golds, preds):
      print idx
      print sent
      print token_sent
      print 'Human label      :\t%s' % ' | '.join(gold)
      print 'Test prediction  :\t%s' % ' | '.join(pred)
    print ''
    print df
    sys.stdout = sys.__stdout__

  exit(1)


def read(file_path):
  i = 0
  idx, sent, token_sent, gold, pred = None, None, None, None, None
  res = []
  now_in_examples = True
  stats = []
  for l in open(file_path):
    l = l.replace('\n', '')
    if now_in_examples:
      m = re.search('^<(\d+)>', l)
      m2 = re.search('^Test input\s*:\s*(.+)', l)
      m3 = re.search('^Test input \(unk\)\s*:\s*(.+)', l)
      m4 = re.search('^Human label\s*:\s*(.+)', l)
      m5 = re.search('^Test prediction\s*:\s*(.+)', l)
      if m:
        idx = m.group(0)
      if m2: 
        sent = m2.group(0)
      if m3: 
        token_sent = m3.group(0)
      if m4:
        gold = [x.strip() for x in m4.group(1).split('|')]
        gold = gold[0]
      if m5:
        pred = [x.strip() for x in m5.group(1).split('|')]
        pred = pred[0]
      if idx and sent and token_sent and gold and pred:
        res.append([idx, sent,token_sent, gold, pred])
        idx, sent, token_sent, gold, pred = None, None, None, None, None
      if not l.strip():
        now_in_examples = False
    else:
      stats.append(l.strip())

  col_name = stats[0]
  pattern = '[0-9\.]+'
  em = re.search(pattern, stats[2]).group(0)
  prec = re.search(pattern, stats[3]).group(0)
  recall = re.search(pattern, stats[4]).group(0)
  indices, sents, token_sents, golds, preds = zip(*res)
  col_stat = [col_name, em, prec, recall]
  return indices, sents, token_sents, golds, preds, col_stat


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("root_path")
  #parser.add_argument('--result_file', default='test.annotated.csv.best.overall')
  parser.add_argument('--result_file', default='test.annotated.csv.best.rate')
  args  = parser.parse_args()
  main(args)


