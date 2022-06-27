###############################################################################


def load_with_verbose_info(filename):
    x, y, n_features, n_classes = load_choirdat(filename)
    print(
        "{}\t{} samples\t{} features\t{} classes".format(
            os.path.basename(filename), x.shape[0], n_features, n_classes
        )
    )
    return x, y, n_classes


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

