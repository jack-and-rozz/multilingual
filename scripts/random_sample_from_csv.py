# coding: utf-8
import argparse, re, sys, os, random
import pandas as pd
random.seed(0)
def main(args):
  df = pd.read_csv(args.file_path)
  res = random.sample(df.values.tolist(), 833)
  res = list(zip(*res))
  new_df = pd.DataFrame()
  for col, val in zip(df, res):
    new_df[col] = val
  new_df = new_df.set_index('index')
  print new_df.to_csv()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file_path")
  args  = parser.parse_args()
  main(args)


