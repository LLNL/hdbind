from sklearn.metrics import classification_report, accuracy_score

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='path to input file containing preds and ground truth')

args = parser.parse_args()


import pandas as pd
def main():

	df = pd.read_csv(args.i, index_col=0)

	# default value is 2 which corresponds to "ambiguous binder" and makes the rest of the code easier
	df['pred_class'] = [2] * len(df)
	df['true_class'] = [2] * len(df)

	df.loc[df['y_pred'] > 8, 'pred_class'] = 1
	df.loc[df['y_pred'] < 6, 'pred_class'] = 0

	df.loc[df['y_true'] > 8, 'true_class'] = 1
	df.loc[df['y_true'] < 6, 'true_class'] = 0

	print(classification_report(y_true=df['true_class'], y_pred=df['pred_class'], labels=[0,1]))
	print(accuracy_score(y_true=df['true_class'], y_pred=df['pred_class']))

if __name__ == '__main__':
	main()