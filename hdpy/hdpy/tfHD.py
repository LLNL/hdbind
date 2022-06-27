import sys
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "10"
import argparse
import time
import random
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
import multiprocessing as mp
#disable_eager_execution()
# tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier

from hdpy.encode_utils import encode
from hdpy.utils import timing_part


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data", type=str, help="choir training dataset")

    parser.add_argument("--test-data", type=str, help="choir testing dataset")

    parser.add_argument(
        "-d",
        "--dimensions",
        default=10000,
        type=int,
        required=False,
        help="sets dimensions value",
        dest="dimensions",
    )

    parser.add_argument(
        "-q",
        "--quantization",
        default=10,
        type=int,
        required=False,
        help="sets quantization level",
        dest="Q",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        default=32,
        type=int,
        required=False,
        help="sets batch size",
        dest="batch_size",
    )

    parser.add_argument(
        "-i",
        "--iterations",
        default=20,
        type=int,
        required=False,
        help="sets iteration number",
        dest="iterations",
    )

    parser.add_argument(
        "-s",
        "--sim_metric",
        default="cos",
        type=str,
        required=False,
        help="sets similarity metric",
        dest="sim_metric",
        choices=["dot", "cos"],
    )

    parser.add_argument(
        "-r",
        "--random-seed",
        default=0,
        type=int,
        required=False,
        help="sets random seed",
        dest="random_seed",
    )

    parser.add_argument(
        "--n-samples",
        default=10,
        type=int,
        required=True,
        help="number of random hyperparam samples to take"
    )

    parser.add_argument(
        "-O",
        "--output-hyperopt-data-csv",
        required=True,
        help="path to csv file where the output of hyperparam optimization will be stored",
        dest="output_hyperopt_data_csv"
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="prefix to use for output files",
        dest="output_prefix"
    )
    parser.add_argument(
        "--tf-device-str",
        required=True,
        help="the tensorflow device id string to use",
        choices=["/CPU:0", "/GPU:0", "/GPU:1"],
    )

    parsed = parser.parse_args()
    return parsed



###############################################################################
## Preproecssing
def normalize(x, x_test=None):
    from sklearn.preprocessing import MinMaxScaler

    if x_test is None:
        x_data = x
    else:
        x_data = np.concatenate((x, x_test), axis=0)

    # PAY ATTENTION TO THIS!!!
    scaler = MinMaxScaler().fit(x_data)
    x_norm = scaler.transform(x)
    if x_test is None:
        return x_norm, None
    else:
        return x_norm, scaler.transform(x_test)


def shuffle_data(x, y):
    sample_order = list(range(x.shape[0]))
    random.shuffle(sample_order)
    return x[sample_order], y[sample_order]


###############################################################################


def singlepass_training(x_h, y, K, tf_device_str:str, return_time:bool):
    with tf.device(tf_device_str):
        with timing_part(str(singlepass_training)) as timing_context:
            model = []
            for c in range(K):
                t = tf.boolean_mask(x_h, y == c)
                model.append(tf.reduce_sum(t, 0))
            
            if return_time:
                return tf.stack(model), timing_context.total_time
            else:
                return tf.stack(model)


def retrain(model, x_h, y, K, x_test_h, y_test, epochs, tf_device_str:str):
    train_accuracy = None
    test_accuracy = None
    for epoch in range(epochs):
        if train_accuracy is not None:
            if test_accuracy is None:
                pbar.set_description("ACC=%.4f" % (train_accuracy))
            else:
                pbar.set_description(
                    "ACC=%.4f TACC=%.4f" % (train_accuracy, test_accuracy)
                )

        n_correct = 0
        N = tf.shape(x_h)[0]
        with timing_part(str(retrain)):
            with tf.device(tf_device_str):
                for i, _y in enumerate(y):
                    sims = tf.linalg.matvec(model, x_h[i]) / tf.norm(
                        model
                    )  # tf.norm(x_h[i])
                    if args.sim_metric == "cos":
                        sims /= tf.norm(model)  # * tf.norm(x_h[i])

                    y_pred = tf.math.argmax(sims)
                    if y_pred == _y:
                        n_correct += 1
                    else:
                        mlist = tf.unstack(model)
                        mlist[y_pred] -= x_h[i]
                        mlist[_y] += x_h[i]
                        model = tf.stack(mlist)

    return model


def retrain_batch(model, x_h, y, K, x_test_h, y_test, epochs, batch_size, sim_metric, tf_device_str:str, return_time:bool):
    B = batch_size

    train_accuracy = None
    test_accuracy = None

    train_time_list = []

    for epoch in range(epochs):
        if train_accuracy is not None:
            if test_accuracy is None:
                pbar.set_description("ACC=%.4f" % (train_accuracy))
            else:
                pbar.set_description(
                    "ACC=%.4f TACC=%.4f" % (train_accuracy, test_accuracy)
                )

        n_correct = 0
        N = tf.shape(x_h)[0]
        timing_context = timing_part(str(retrain))
        with timing_context:
            with tf.device(tf_device_str):
                for i in range(0, N, B):
                    b = min(B, N - i)
                    x_batch = tf.slice(x_h, [i, 0], [b, -1])
                    y_batch = y[i : i + B]

                    sims = tf.linalg.matmul(model, x_batch, False, True)
                    if sim_metric == "cos":
                        sims /= tf.norm(model)  # * tf.norm(x_h[i])
                    y_pred = tf.math.argmax(sims, 0)
                    wrong = y_pred != y_batch  # .numpy()

                    mlist = tf.unstack(model)
                    for c in range(K):
                        corct_mask = wrong & (y_batch == c)
                        wrong_mask = wrong & (y_pred == c)

                        # add the HVs of the correct predictions
                        mlist[c] += tf.reduce_sum(
                            tf.boolean_mask(x_batch, corct_mask), 0
                        )
                        # subtract the HVs of the incorrect predictions
                        mlist[c] -= tf.reduce_sum(
                            tf.boolean_mask(x_batch, wrong_mask), 0
                        )
                    model = tf.stack(mlist)


                train_time_list.append(timing_context.total_time)

    if return_time:
        return model, np.sum(train_time_list)
    else:
        return model


def test(model, x_test_h, y_test, sim_metric, return_time:bool):
    with timing_part(str(test)) as timing_context:
        sims = tf.linalg.matmul(x_test_h, model, False, True) / tf.norm(model)

        if sim_metric == "cos":
            sims /= tf.norm(model)  # * tf.norm(x_h[i])
        y_pred = tf.argmax(sims, 1)
        n_correct = tf.reduce_sum(tf.cast(y_test == y_pred, tf.float32))

    if return_time:
        return y_pred, y_test, timing_context.total_time
    else:
        return y_pred, y_test



def train_test_loop(
    x_train,
    x_test,
    y_train,
    y_test,
    iterations,
    dimensions,
    Q,
    K,
    batch_size,
    sim_metric,
    tf_device_str:str,
): 

    # Encoding
    train_test_encodings, train_encode_time, test_encode_time = encode(x_train, x_test, dimensions, Q, batch_size, tf_device_str=tf_device_str, return_time=True)
    x_train_h, x_test_h = train_test_encodings
    
    # Training
    model, train_time = singlepass_training(x_train_h, y_train, K, tf_device_str=tf_device_str, return_time=True)
    sp_y_pred, sp_y_true, sp_test_time = test(model, x_test_h, y_test, sim_metric, return_time=True)
    print("test time sp:", sp_test_time)
    model, retrain_time = retrain_batch(
        model,
        x_train_h,
        y_train,
        K,
        x_test_h,
        y_test,
        iterations,
        batch_size,
        sim_metric,
        tf_device_str=tf_device_str,
        return_time=True,
    )
    y_pred, y_true, retrain_test_time = test(model, x_test_h, y_test, sim_metric, return_time=True)

    return model, y_pred, y_true, train_time, retrain_time, sp_test_time, retrain_test_time, train_encode_time, test_encode_time

def optimize_HD_model(x_train, x_test, y_train, y_test,K:int, n_samples:int, batch_size:int, output_hyperopt_data_csv:str,
                    tf_device_str:str):



    result_list = []

    encoding_dim_dist = [2, 10, 100, 1000, 10000]
    quantization_level_dist = [1, 2, 4, 8, 16, 32]
    sim_metric_dist = ['cos', None]

    iteration_list = [1, 10, 100]

    combos = list(itertools.product(encoding_dim_dist, quantization_level_dist, sim_metric_dist, iteration_list))

    tqdm.write(f"{len(combos)} trials to run..")

    best_idx = 0
    best_f1 = -999

    hyperopt_dict = {"iterations": [],
                    "encoding_dim": [],
                    "quantization_level": [],
                    "sim_metric": [],
                    "accuracy_score": [],
                    "precision_score": [],
                    "recall_score": [],
                    "f1_score": [],
                    "train_time": [],
                    "retrain_time": [],
                    "singlepass_test_time": [],
                    "retrain_test_time": [],
                    "train_encode_time": [],
                    "test_encode_time": []}

    for idx, hyperopt_combo in enumerate(tqdm(combos, total=len(combos))):

        encoding_dim_num, quantization_level_num, sim_metric, iterations = hyperopt_combo

        model, y_test_pred, y_test_true, train_time, retrain_time, sp_test_time, retrain_test_time, train_encode_time, test_encode_time = train_test_loop(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                iterations=iterations, dimensions=encoding_dim_num,
                Q=quantization_level_num, K=K, batch_size=batch_size, sim_metric=sim_metric, tf_device_str=tf_device_str)


        f1 = f1_score(y_pred=y_test_pred, y_true=y_test_true)
        result_list.append((model, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_idx = idx


        prec = precision_score(y_pred=y_test_pred, y_true=y_test_true)
        rec = recall_score(y_pred=y_test_pred, y_true=y_test_true)
        acc = accuracy_score(y_pred=y_test_pred, y_true=y_test_true)

        hyperopt_dict["iterations"].append(iterations)
        hyperopt_dict["encoding_dim"].append(encoding_dim_num)
        hyperopt_dict["quantization_level"].append(quantization_level_num)
        hyperopt_dict["sim_metric"].append(sim_metric)
        hyperopt_dict["accuracy_score"].append(acc)
        hyperopt_dict["f1_score"].append(f1)
        hyperopt_dict["precision_score"].append(prec)
        hyperopt_dict["recall_score"].append(rec)
        hyperopt_dict["train_time"].append(train_time)
        hyperopt_dict["retrain_time"].append(retrain_time)
        hyperopt_dict["singlepass_test_time"].append(sp_test_time)
        hyperopt_dict["retrain_test_time"].append(retrain_test_time)
        hyperopt_dict["train_encode_time"].append(train_encode_time)
        hyperopt_dict["test_encode_time"].append(test_encode_time)

        print(f"iterations: {iterations}, batch_size: {batch_size}, encoding_dim_num: {encoding_dim_num},"\
        f"quantization_level_num: {quantization_level_num}, sim_metric: {sim_metric}, f1: {f1}")


    print(f"best f1: {best_f1}")
    return result_list[best_idx], hyperopt_dict


def main(): 

    best_model = optimize_HD_model(x_train, x_test, y_train, y_test, K=K, n_samples=args.n_samples, iterations=args.iterations, batch_size=args.batch_size,
                                    output_hyperopt_data_csv=args.output_hyperopt_data_csv, tf_device_str=args.tf_device_str)
 

if __name__ == "__main__":
    args = parse_args()

    # Random seed initialization
    print("Random seed: {}".format(args.random_seed))
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    # File Loading
    x_train, y_train, K = load_with_verbose_info(args.train_data)

    x_test, y_test, _ = load_with_verbose_info(args.test_data)
    n_test = len(x_test)


    # Preprocessing
    x_train, x_test = normalize(x_train, x_test)

    x_train, y_train = shuffle_data(
        x_train, y_train
    )  # need to shuffle training set only

    main()
