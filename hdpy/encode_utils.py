################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
from symbol import import_from
import numpy as np
import tensorflow as tf

import multiprocessing as mp
from hdpy.utils import timing_part


# Base hypervectors
def create_random_hypervector(D):
    """
    the code below create a hypervector of 1s, checks that the size of the hypervector `D` is divisible by 2,
    then assigns half the vector to the value -1, then randomly permutes these values thus creating a random hypervector
    """
    hv = np.ones(D, dtype=np.float32)
    assert D % 2 == 0
    hv[D // 2 :] = -1.0
    return hv[np.random.permutation(D)]


# what is the purpose of the code below? is this padding an array to conform to the expected batch_size?
# if so then this should be called padding, this isn't clear
def create_dummy_for_batch(x, batch_size):
    N = x.shape[0]
    dummy_size = batch_size - N % batch_size
    if dummy_size == batch_size:
        return x

    dummy = np.zeros((dummy_size, x.shape[1]))
    return np.vstack((x, dummy))


# 2. Encode with Tensorflow
# Note: The performance of TF is largely dependent upon the implementation
# This version (maybe final one) shows the best perf with scalability
# among multiple different variants tried
def encode_tf(
    N,
    Q,
    level_hvs,
    id_hvs,
    feature_matrix,
    batch_size,
    tf_device_str: str,
    return_time: bool,
    verbose=False,
):
    assert N % batch_size == 0

    with tf.device(tf_device_str):
        # encode_sample is the encoding method
        """
        tf.reduce_sum is being applied over the axis=1 dimension, so level_hvs has shape N by encoding_dim

        tf.scalar_mul(
                        Q, tf.gather(feature_matrix, tf.range(i, i + batch_size))
                    ),

        gathers a batch from the feature_matrix, then multiplies the elements by Q

        tf.cast converts the result to another type, in this case tf.int32, which for floats should just take the integer part of the number

        tf.cast(tf.scalar_mul(Q, tf.gather(feature_matrix, tf.range(i, i + batch_size))), tf.int32,)
        appears to be computing a set of indices that are used to "gather" vectors from level_hvs, i.e.
        its computing which quantization level each hypervector belongs to

        i.e. level_hvs is a map from quantization level to what are random hypervectors...

        the outer level call of tf.reduce_sum appears to be summing over the feature_dimension...i.e. the resulting array from the
        inner-most logic produces a tensor of shape BATCH_SIZE by FEATURE_SIZE by ENCODING_DIM, and after mapping the features(?) to a
        quantization level HV, sums up the elements over the feature_dimension, producing a matrix of shape BATCH_SIZE by ENCODING DIM

        does this make sense to do for the dataset?
        """

        encode_sample = lambda i: tf.reduce_sum(
            input_tensor=tf.multiply(
                tf.gather(
                    level_hvs,
                    tf.cast(
                        tf.scalar_mul(
                            Q, tf.gather(feature_matrix, tf.range(i, i + batch_size))
                        ),
                        tf.int32,
                    ),
                ),
                id_hvs,
            ),
            axis=1,
        )

        # N // batch_size is computing the number of partitions of data of size batch_size
        tf_hv_matrix = tf.TensorArray(dtype=tf.float32, size=N // batch_size)
        # cond appears to be the condition that the while loop uses to iterate until
        cond = lambda i, _: i < N
        # body is the body of the while loop, ta(?)
        body = lambda i, ta: (
            i + batch_size,
            ta.write(i // batch_size, encode_sample(i)),
        )

        try:
            with timing_part(str(encode_tf), verbose=verbose) as timing_context:
                tf_hv_matrix_final = tf.while_loop(
                    cond=cond, body=body, loop_vars=(0, tf_hv_matrix)
                )[1]
                # writer = tf.summary.FileWriter('./graphs', sess.graph)

                encodings = tf.reshape(
                    tf_hv_matrix_final.stack(),
                    (feature_matrix.shape[0], level_hvs.shape[1]),
                )

            if return_time:
                return encodings, timing_context.total_time

            else:
                return encodings
        except Exception as e:
            print(e)


def encode_batch(x, D, Q, batch_size, tf_device_str: str, return_time: bool):
    # F is the size of the feature dimension
    F = x.shape[1]

    # base hypervectors
    # TODO (derek): what does "level_base" mean?
    # TODO (derek): what does "id_base" mean?
    level_base = create_random_hypervector(D)
    id_base = create_random_hypervector(D)

    # the code below appears to be doing "quantization"
    # level_hvs will have shape args.Q + 1 by encoding_dimension
    level_hvs = []
    for q in range(Q + 1):
        flip = int(q / Q)
        level_hv = np.copy(level_base)
        level_hv[:flip] = level_base[:flip] * -1.0
        level_hvs.append(level_hv[np.random.permutation(D)])
    level_hvs = np.array(level_hvs, dtype=np.float32)

    """
    the code below is shifiting the array in a circular direction (where end element becomes first element and all others shift either
    right if `f` is positive or to the left if `f` is negative)
    """
    id_hvs = []
    for f in range(F):
        id_hvs.append(np.roll(id_base, f))
    id_hvs = np.array(id_hvs, dtype=np.float32)

    # pad x_train and x_test
    x = create_dummy_for_batch(x, batch_size)

    x_h = encode_tf(
        x.shape[0],
        Q,
        level_hvs,
        id_hvs,
        x,
        batch_size,
        tf_device_str=tf_device_str,
        return_time=return_time,
    )

    # if return_time=True, then encode_tf will return two values so we have to unpack those tuples
    if return_time:
        x_h, encode_time = x_h

    # train_test_encodings = tf.slice(x_train_h, [0, 0], [N, -1]), tf.slice(x_test_h, [0, 0], [N_test, -1])

    encodings = x_h

    if return_time:
        return encodings, encode_time
    else:
        return encodings


# TODO (derek): rename this function as encode_train_test
def encode(x_train, x_test, D, Q, batch_size, tf_device_str: str, return_time: bool):
    # F is the size of the feature dimension
    F = x_train.shape[1]

    # base hypervectors
    # TODO (derek): what does "level_base" mean?
    # TODO (derek): what does "id_base" mean?
    level_base = create_random_hypervector(D)
    id_base = create_random_hypervector(D)

    # the code below appears to be doing "quantization"
    # level_hvs will have shape args.Q + 1 by encoding_dimension
    level_hvs = []
    for q in range(Q + 1):
        flip = int(q / Q)
        level_hv = np.copy(level_base)
        level_hv[:flip] = level_base[:flip] * -1.0
        level_hvs.append(level_hv[np.random.permutation(D)])
    level_hvs = np.array(level_hvs, dtype=np.float32)

    """
    the code below is shifiting the array in a circular direction (where end element becomes first element and all others shift either
    right if `f` is positive or to the left if `f` is negative)
    """
    id_hvs = []
    for f in range(F):
        id_hvs.append(np.roll(id_base, f))
    id_hvs = np.array(id_hvs, dtype=np.float32)

    # pad x_train and x_test
    N = x_train.shape[0]
    x_train = create_dummy_for_batch(x_train, batch_size)
    N_test = x_test.shape[0]
    x_test = create_dummy_for_batch(x_test, batch_size)

    x_train_h = encode_tf(
        x_train.shape[0],
        Q,
        level_hvs,
        id_hvs,
        x_train,
        batch_size,
        tf_device_str=tf_device_str,
        return_time=return_time,
    )
    x_test_h = encode_tf(
        x_test.shape[0],
        Q,
        level_hvs,
        id_hvs,
        x_test,
        batch_size,
        tf_device_str=tf_device_str,
        return_time=return_time,
    )

    # if return_time=True, then encode_tf will return two values so we have to unpack those tuples
    if return_time:
        x_train_h, train_encode_time = x_train_h
        x_test_h, test_encode_time = x_test_h

    train_test_encodings = tf.slice(x_train_h, [0, 0], [N, -1]), tf.slice(
        x_test_h, [0, 0], [N_test, -1]
    )

    if return_time:
        return train_test_encodings, train_encode_time, test_encode_time
    else:
        return train_test_encodings


def encode_single(
    x, D, Q, batch_size, tf_device_str: str, return_time: bool, verbose=False
):
    # F is the size of the feature dimension
    F = x.shape[1]

    # base hypervectors
    # TODO (derek): what does "level_base" mean?
    # TODO (derek): what does "id_base" mean?
    level_base = create_random_hypervector(D)
    id_base = create_random_hypervector(D)

    # the code below appears to be doing "quantization"
    # level_hvs will have shape args.Q + 1 by encoding_dimension
    level_hvs = []
    for q in range(Q + 1):
        flip = int(q / Q)
        level_hv = np.copy(level_base)
        level_hv[:flip] = level_base[:flip] * -1.0
        level_hvs.append(level_hv[np.random.permutation(D)])
    level_hvs = np.array(level_hvs, dtype=np.float32)

    """
    the code below is shifiting the array in a circular direction (where end element becomes first element and all others shift either
    right if `f` is positive or to the left if `f` is negative)
    """
    id_hvs = []
    for f in range(F):
        id_hvs.append(np.roll(id_base, f))
    id_hvs = np.array(id_hvs, dtype=np.float32)

    # pad x_train and x_test
    N = x.shape[0]
    x = create_dummy_for_batch(x, batch_size)

    x_h = encode_tf(
        x.shape[0],
        Q,
        level_hvs,
        id_hvs,
        x,
        batch_size,
        tf_device_str=tf_device_str,
        return_time=return_time,
        verbose=verbose,
    )

    # if return_time=True, then encode_tf will return two values so we have to unpack those tuples
    if return_time:
        x_h, encode_time = x_h

    if return_time:
        return x_h, encode_time
    else:
        return x_h
