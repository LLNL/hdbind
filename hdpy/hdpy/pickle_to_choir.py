import pickle
import numpy
import struct

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="path that points to data pickle file", default='eeg1024.pickle')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--output-dir', help="path to output directory")

    args = parser.parse_args()

    return args

def writeDataSetForChoirSIM(ds, filename):
    X, y = ds
    f = open(filename, "wb")

    # import pdb
    # pdb.set_trace()
    nFeatures = len(X[0])
    nClasses = len(set(y))

    f.write(struct.pack('i', nFeatures))
    f.write(struct.pack('i', nClasses))
    for V, l in zip(X, y):
        for v in V:
            # import pdb
            # pdb.set_trace()
            f.write(struct.pack('f', v))
        f.write(struct.pack('i', l))

def main(args):
    from pathlib import Path
    data_path = Path(args.data)

    if args.test_only:
        with open(data_path, 'rb') as f:
            X_test, y_test = pickle.load(f)

        print(len(X_test))
        print(len(y_test))

        output_prefix = data_path.name.split('.')[0]
        output_path = Path(f"{args.output_dir}/")
        if not output_path.exists():
            output_path.mkdir(exist_ok=True)

        writeDataSetForChoirSIM((X_test, y_test), f"{output_path}/{output_prefix}_test.choir_dat")

    else:
        with open(data_path, 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)

        print(len(X_train))
        print(len(y_train))
        print(len(X_test))
        print(len(y_test))

        output_prefix = data_path.name.split('.')[0]
        output_path = Path(f"{args.output_dir}/")
        if not output_path.exists():
            output_path.mkdir(exist_ok=True)

        writeDataSetForChoirSIM((X_train, y_train), f"{output_path}/{output_prefix}_train.choir_dat")
        writeDataSetForChoirSIM((X_test, y_test), f"{output_path}/{output_prefix}_test.choir_dat")




if __name__ == '__main__':
    args = get_args()
    main(args)