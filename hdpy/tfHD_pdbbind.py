"""
This is a training script geared specifically for PDBBind where the user will be able to 
specify the cutoff thresholds for the classes on demand.
"""
import sys
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "10"
import argparse
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
import multiprocessing as mp
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.dummy import DummyClassifier
from hdpy.tfHD import (
    parse_args,
    normalize,
    shuffle_data,
    optimize_HD_model,
)
from hdpy.encode_utils import encode
from hdpy.utils import timing_part, convert_real_to_int_label

def load_numpy_data_with_real_label(data_path, binary_only=False):
    data = np.load(data_path)
    x = data[:, :-1]
    y = data[:, -1]

    y = np.asarray([convert_real_to_int_label(x) for x in y])

    if binary_only:
        binary_label_mask = y[:] != 2
        return x[binary_label_mask, :], y[binary_label_mask]
    else:
        return x, y


def main():

    pass


if __name__ == "__main__":
    args = parse_args()

    # Random seed initialization
    print("Random seed: {}".format(args.random_seed))
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    x_train, y_train = load_numpy_data_with_real_label(
        args.train_data, binary_only=True
    )

    x_test, y_test = load_numpy_data_with_real_label(args.test_data, binary_only=True)

    n_test = len(x_test)

    # Preprocessing
    x_train, x_test = normalize(x_train, x_test)

    x_train, y_train = shuffle_data(
        x_train, y_train
    )  # need to shuffle training set only


    best_model, hyperopt_dict = optimize_HD_model(x_train=x_train, x_test=x_test, 
                y_train=y_train, y_test=y_test,K=2, n_samples=args.n_samples,
                batch_size=args.batch_size, 
                output_hyperopt_data_csv=args.output_hyperopt_data_csv, tf_device_str=args.tf_device_str)

    import pickle

    from pathlib import Path

    output_csv_path = Path(args.output_hyperopt_data_csv)

    hyperopt_df = pd.DataFrame(hyperopt_dict)
    if not output_csv_path.parent.exists():
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    hyperopt_df.to_csv(output_csv_path)
    output_best_model_path = output_csv_path.with_name(f"{args.output_prefix}_best_model.pkl")

    with open(output_best_model_path, 'wb') as handle:
        pickle.dump(best_model, handle) 