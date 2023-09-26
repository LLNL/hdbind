import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm


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

    f = h5py.File(hdf_path, "r")

    # need an argument for train split, need an argument for test split

    train_df = pd.read_csv(train_split_path)
    test_df = pd.read_csv(test_split_path)

    train_ids = train_df["pdbid"].values.tolist()
    test_ids = test_df["pdbid"].values.tolist()

    train_list = []
    test_list = []

    for key in tqdm(list(f), total=len(list(f))):
        if key in train_ids:
            train_list.append(key)
        elif key in test_ids:
            test_list.append(key)
        else:
            print(f"key: {key} not contained in train or test split")
            continue

    train_data_list = []
    train_label_list = []
    for key in train_list:
        affinity = f[key].attrs['affinity']

        if affinity > bind_thresh:
            train_label_list.append(1)
        elif affinity < no_bind_thresh:
            train_label_list.append(0)
        else:

            print(f"key: {key} has ambiguous label")
            continue

        
        train_data_list.append(np.asarray(f[key][dataset_name]))
            

    test_data_list = []
    test_label_list = []
    for key in test_list:
        affinity = f[key].attrs['affinity']

        if affinity > bind_thresh:
            test_label_list.append(1)
        elif affinity < no_bind_thresh:
            test_label_list.append(0)
        else:

            print(f"key: {key} has ambiguous label")
            continue

        test_data_list.append(np.asarray(f[key][dataset_name]))

    return (np.asarray(train_data_list), np.asarray(train_label_list), np.asarray(test_data_list), np.asarray(test_label_list))
def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf-path', help='path to input hdf file containing pdbbind data')    
    parser.add_argument('--dataset-name', nargs='+', help='sequence of dataset names to use from the hdf-path dataset')
    parser.add_argument('--train-split-path', help='path to list of pdbids corresponding to the training set')
    parser.add_argument('--test-split-path', help='path to list of pdbids corresponding to the testing set')
    parser.add_argument('--bind-thresh', type=float, help='threshold (lower) to use for determining binders from experimental measurement')
    parser.add_argument('--no-bind-thresh', type=float, help='threshold (upper) to use for determining non-binders from experimental measurement')
    parser.add_argument('--run-HD-benchmark', action='store_true')
    args = parser.parse_args()

    for dataset_name in args.dataset_name:
        data = load_pdbbind_from_hdf(hdf_path=args.hdf_path, dataset_name=dataset_name, 
                    train_split_path=args.train_split_path,
                    test_split_path=args.test_split_path,
                    bind_thresh=args.bind_thresh,
                    no_bind_thresh=args.no_bind_thresh)

        x_train, y_train, x_test, y_test = data

        #import pdb
        #pdb.set_trace()
        if args.run_HD_benchmark:
            from hdpy.tfHD import train_test_loop

            train_test_loop(x_train.squeeze(), x_test.squeeze(), y_train, y_test, iterations=10, dimensions=10000, Q=10, K=2, batch_size=32, sim_metric='cos')

    print(data)



def convert_pdbbind_affinity_to_class_label(x, pos_thresh=8, neg_thresh=6):


    if x < neg_thresh:
        return 0
    elif x > pos_thresh:
        return 1
    else:
        return 2

if __name__ == '__main__':
    main()
