import sys
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "10"
import argparse
import time
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()
# tf.compat.v1.disable_eager_execution()
# tf.executing_eagerly()
from tqdm import tqdm


def timing(func):
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        print(str(func) + "- " + str(time.time() - ts))
        return result

    return wrapper


class timing_part:
    def __init__(self, TAG):
        self.TAG = str(TAG)

    def __enter__(self):
        self.ts = time.time()

    def __exit__(self, type, value, traceback):
        print(str(self.TAG) + "\t" + str(time.time() - self.ts))


###############################################################################
## Loads .choirdat file's information
# returns a tuple: ((samples, labels), features, classes); where features and
# classes is a number and samples and labels is a list of vectors
def load_choirdat(dataset_path, use_pickle=True):
    import hashlib
    import pickle

    def get_file_hash(dataset_path):
        with open(dataset_path, "rb") as f:
            m = hashlib.sha256()
            m.update(f.read())
        return m.hexdigest()[:16]

    PICKLE_DIR = "tfHD.cache"
    if use_pickle:
        pickle_filepath = PICKLE_DIR + "/" + get_file_hash(dataset_path)
        if not os.path.exists(PICKLE_DIR):
            os.mkdir(PICKLE_DIR)

        if os.path.exists(pickle_filepath):
            with open(pickle_filepath, "rb") as f:
                return pickle.load(f)

    def return_with_pickle(*ret):
        if use_pickle:
            with open(pickle_filepath, "wb") as f:
                pickle.dump(ret, f)
        return ret

    import struct

    with open(dataset_path, "rb") as f:
        # reads meta information
        features = struct.unpack("i", f.read(4))[0]
        classes = struct.unpack("i", f.read(4))[0]

        # lists containing all samples and labels to be returned
        samples = list()
        labels = list()

        while True:
            # load a new sample
            sample = list()

            # load sample's features
            for i in range(features):
                val = f.read(4)
                if val is None or not len(val):
                    return return_with_pickle(
                        np.array(samples), np.array(labels), features, classes
                    )
                sample.append(struct.unpack("f", val)[0])

            # add the new sample and its label
            label = struct.unpack("i", f.read(4))[0]
            samples.append(sample)
            labels.append(label)

    return return_with_pickle(
        np.array(samples, np.float32), np.array(labels, np.int32), features, classes
    )


###############################################################################

###############################################################################
## Preproecssing
def normalize(x, x_test=None):
    from sklearn.preprocessing import MinMaxScaler

    if x_test is None:
        x_data = x
    else:
        x_data = np.concatenate((x, x_test), axis=0)

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

###############################################################################


def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('filename', type=str,
    #        help='choir training dataset')

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
        "--batchsize",
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
        "--randomseed",
        default=0,
        type=int,
        required=False,
        help="sets random seed",
        dest="random_seed",
    )

    parsed = parser.parse_args()
    return parsed


def load_with_verbose_info(filename):
    x, y, n_features, n_classes = load_choirdat(filename)
    print(
        "{}\t{} samples\t{} features\t{} classes".format(
            os.path.basename(filename), x.shape[0], n_features, n_classes
        )
    )
    return x, y, n_classes


###############################################################################

# 2. Encode with Tensorflow
# Note: The performance of TF is largely dependent upon the implementation
# This version (maybe final one) shows the best perf with scalability
# among multiple different variants tried
def encode_tf(N, Q, level_hvs, id_hvs, feature_matrix):
    global args
    BATCH_SIZE = args.batch_size
    assert N % BATCH_SIZE == 0

    with tf.device("/gpu:0"):
        encode_sample = lambda i: tf.reduce_sum(
            input_tensor=tf.multiply(
                tf.gather(
                    level_hvs,
                    tf.cast(
                        tf.scalar_mul(
                            Q,
                            tf.gather(feature_matrix, tf.range(i, i + BATCH_SIZE))
                            # tf.convert_to_tensor(feature_matrix[i:i+BATCH_SIZE])
                        ),
                        tf.int32,
                    ),
                ),
                id_hvs,
            ),
            axis=1,
        )

        tf_hv_matrix = tf.TensorArray(dtype=tf.float32, size=N // BATCH_SIZE)
        cond = lambda i, _: i < N
        body = lambda i, ta: (
            i + BATCH_SIZE,
            ta.write(i // BATCH_SIZE, encode_sample(i)),
        )
        with timing_part(str(encode_tf)):
            tf_hv_matrix_final = tf.while_loop(
                cond=cond, body=body, loop_vars=(0, tf_hv_matrix)
            )[1]
            # writer = tf.summary.FileWriter('./graphs', sess.graph)
            return tf.reshape(
                tf_hv_matrix_final.stack(),
                (feature_matrix.shape[0], level_hvs.shape[1]),
            )


def encode(x, x_test, D, Q):
    global args
    F = x.shape[1]

    # Base hypervectors
    def create_random_hypervector(D):
        hv = np.ones(D, dtype=np.float32)
        assert D % 2 == 0
        hv[D // 2 :] = -1.0
        return hv[np.random.permutation(D)]

    level_base = create_random_hypervector(D)
    id_base = create_random_hypervector(D)

    level_hvs = []
    for q in range(Q + 1):
        flip = int(q / Q)
        level_hv = np.copy(level_base)
        level_hv[:flip] = level_base[:flip] * -1.0
        level_hvs.append(level_hv[np.random.permutation(D)])
    level_hvs = np.array(level_hvs, dtype=np.float32)

    id_hvs = []
    for f in range(F):
        id_hvs.append(np.roll(id_base, f))
    id_hvs = np.array(id_hvs, dtype=np.float32)

    BATCH_SIZE = args.batch_size

    def create_dummy_for_batch(x, BATCH_SIZE):
        N = x.shape[0]
        dummy_size = BATCH_SIZE - N % BATCH_SIZE
        if dummy_size == BATCH_SIZE:
            return x

        dummy = np.zeros((dummy_size, x.shape[1]))
        return np.vstack((x, dummy))

    N = x.shape[0]
    x = create_dummy_for_batch(x, BATCH_SIZE)
    N_test = x_test.shape[0]
    x_test = create_dummy_for_batch(x_test, BATCH_SIZE)

    x_h = encode_tf(x.shape[0], Q, level_hvs, id_hvs, x)
    x_test_h = encode_tf(x_test.shape[0], Q, level_hvs, id_hvs, x_test)

    # print(tf.shape(x_h))
    # print(tf.shape(x_test_h))

    return tf.slice(x_h, [0, 0], [N, -1]), tf.slice(x_test_h, [0, 0], [N_test, -1])


def singlepass_training(x_h, y, K):
    with tf.device("/gpu:0"):
        with timing_part(str(singlepass_training)):
            model = []
            for c in range(K):
                t = tf.boolean_mask(x_h, y == c)
                model.append(tf.reduce_sum(t, 0))
            return tf.stack(model)


def retrain(model, x_h, y, K, x_test_h, y_test, epochs):
    pbar = tqdm(range(epochs))
    train_accuracy = None
    test_accuracy = None
    for epoch in pbar:
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
            with tf.device("/gpu:0"):
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


def retrain_batch(model, x_h, y, K, x_test_h, y_test, epochs):
    global args
    B = args.batch_size

    pbar = tqdm(range(epochs))
    train_accuracy = None
    test_accuracy = None
    for epoch in pbar:
        if train_accuracy is not None:
            if test_accuracy is None:
                pbar.set_description("ACC=%.4f" % (train_accuracy))
            else:
                pbar.set_description(
                    "ACC=%.4f TACC=%.4f" % (train_accuracy, test_accuracy)
                )

        n_correct = 0
        # N = tf.shape(x_h)[0]
        N = tf.shape(x_h)[0]
        print(f"N: {N}, B: {B}")
        with timing_part(str(retrain)):
            with tf.device("/gpu:0"):
                for i in range(0, N, B):
                    b = min(B, N - i)
                    x_batch = tf.slice(x_h, [i, 0], [b, -1])
                    y_batch = y[i : i + B]

                    sims = tf.linalg.matmul(model, x_batch, False, True)
                    if args.sim_metric == "cos":
                        sims /= tf.norm(model)  # * tf.norm(x_h[i])
                    y_pred = tf.math.argmax(sims, 0)
                    wrong = y_pred != y_batch  # .numpy()

                    mlist = tf.unstack(model)
                    for c in range(K):
                        corct_mask = wrong & (y_batch == c)
                        wrong_mask = wrong & (y_pred == c)

                        mlist[c] += tf.reduce_sum(
                            tf.boolean_mask(x_batch, corct_mask), 0
                        )
                        mlist[c] -= tf.reduce_sum(
                            tf.boolean_mask(x_batch, wrong_mask), 0
                        )
                    model = tf.stack(mlist)

    return model


from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier


def test(model, x_test_h, y_test):
    with timing_part(str(test)):
        sims = tf.linalg.matmul(x_test_h, model, False, True) / tf.norm(model)
        if args.sim_metric == "cos":
            sims /= tf.norm(model)  # * tf.norm(x_h[i])
        y_pred = tf.argmax(sims, 1)
        n_correct = tf.reduce_sum(tf.cast(y_test == y_pred, tf.float32))

    # print("Accuracy: {}".format(n_correct.numpy() / y_test.shape[0]))

    # print(y_test, y_pred.numpy())

    # print(f"Accuracy: {n_correct / y_test.shape[0]}\nROC_AUC_score: {roc_auc_score(y_pred=y_pred, y_true=y_test)}")
    print(classification_report(y_pred=y_pred, y_true=y_test))


def load_pdbbind_from_hdf(
    hdf_path,
    dataset_name,
    train_split_path,
    test_split_path,
    bind_thresh,
    no_bind_thresh,
):

    """
    input: parameters (dataset name, threshold values, train/test splits) and h5 file containing data and binding measurements
    output: numpy array with features
    """
    import h5py

    f = h5py.File(hdf_path, "r")

    # need an argument for train split, need an argument for test split

    train_df = pd.read_csv(train_split_path)
    test_df = pd.read_csv(test_split_path)

    train_ids = train_df["pdbid"]
    test_ids = train_df["pdbid"]

    train_list = []
    test_list = []

    for key in list(f):
        if key in train_ids:
            train_list.append(key)
        elif key in test_ids:
            test_list.append(key)
        else:
            print(f"key: {key} not contained in train or test split")
            continue

    train_data_list = []
    for pdbid in train_list:
        pass

    test_data_list = []


def main():
    global args
    args = parse_args()

    # Random seed initialization
    # TODO: TF random seed for CUDA if any
    print("Random seed: {}".format(args.random_seed))
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # File Loading
    x, y, K = load_with_verbose_info(args.train_data)

    x_test, y_test, _ = load_with_verbose_info(args.test_data)
    n_test = len(x_test)

    # Preprocessing
    x, x_test = normalize(x, x_test)
    x, y = shuffle_data(x, y)  # need to shuffle training set only

    # Encoding
    print("Encoding: D = {}".format(args.dimensions))
    x_h, x_test_h = encode(x, x_test, args.dimensions, args.Q)
    # print(tf.size(x_h), tf.size(x_test_h))

    # Training
    model = singlepass_training(x_h, y, K)
    test(model, x_test_h, y_test)

    # model = retrain(model, x_h, y, K, x_test_h, y_test, args.iterations)
    model = retrain_batch(model, x_h, y, K, x_test_h, y_test, args.iterations)
    test(model, x_test_h, y_test)

    dummy_class_model_uniform = DummyClassifier(
        strategy="uniform", random_state=args.random_seed
    )
    dummy_class_model_most_frequent = DummyClassifier(
        strategy="most_frequent", random_state=args.random_seed
    )

    dummy_class_model_uniform.fit(x, y)
    dummy_class_model_most_frequent.fit(x, y)

    y_pred_dummy_uniform = dummy_class_model_uniform.predict(x_test)
    y_pred_dummy_most_frequent = dummy_class_model_most_frequent.predict(x_test)

    print("baseline: most frequent label")
    print(classification_report(y_pred=y_pred_dummy_most_frequent, y_true=y_test))
    print("baseline: uniformly equal label")
    print(classification_report(y_pred=y_pred_dummy_uniform, y_true=y_test))


if __name__ == "__main__":
    main()
