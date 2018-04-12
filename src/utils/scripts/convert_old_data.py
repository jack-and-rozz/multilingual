import re, argparse
import pandas as pd

def find_entry(target_labels):
  labels = [x.strip() for x in re.search('\((.+)\)', target_labels.split(':')[0]).group(1).split('|')]
  return labels

def main(args):
  df1 = pd.read_csv(args.data_path)
  labels = [find_entry(x) for x in df1['label'].tolist()]
  lowerbounds, l_equals, upperbounds, u_equals, currencies, rates = zip(*labels)
  df2 = pd.DataFrame({
    'index': df1['index'],
    'sentence': df1['sentence'],
    'LB': lowerbounds,
    'UB': upperbounds,
    'currency': currencies,
    'rate': rates
  }).ix[:, ['index', 'sentence', 'LB', 'UB', 'Currency', 'Rate']]
  df2 = df2.set_index('index')
  print df2.to_csv()
  


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("data_path")
  args  = parser.parse_args()
  main(args)

