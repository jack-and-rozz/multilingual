# coding: utf-8
import pandas as pd
import argparse, sys, os, re

def find_entry(l):
  return [x.strip() for x in re.search('\((.+)\)',l.split(':')[0]).group(1).split('|')]

def main(args):
  df = pd.read_csv(args.file_path)
  new_entries = []
  for l in df.values.tolist():
    lower, _, upper, _, currency, rate = [x.split() for x in find_entry(l[2])]
    keep = False
    if len(lower) > 1 or len(upper) > 1:
      keep = True
    if lower != upper:
      keep = True
    # if currency != ['$'] or rate != ['-']:
    #   keep = True

    if keep == True:
      new_entries.append(l)
      #print l, keep
  idxs, texts, labels = list(zip(*new_entries))
  header = ['index', 'sentence', 'label']
  idx_header, sentence_header, label_header = header
  df = pd.DataFrame({
    idx_header: idxs,
    sentence_header: texts,
    label_header: labels
  }).ix[:, header].set_index(idx_header)
  print df.to_csv()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--file_path", default='dataset/test.annotated.csv')
  args  = parser.parse_args()
  main(args)

