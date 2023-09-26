import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path 
import sys 


def random_partition_dude_data(data_file, output_dir):

    output_train_path = output_dir / "train.npy"
    output_test_path = output_dir / "test.npy"
    print(f"{data_file}\t{output_train_path}\t{output_test_path}")
    data = np.load(data_file)
    label_index = -1
    x_train, x_test = train_test_split(data, stratify=data[:, label_index], test_size=0.2)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_train_path, x_train)
    np.save(output_test_path, x_test)


def partition_lit_pcba_data(data_dir):

    print(data_dir)
    actives_file = data_dir / Path("actives/actives/ecfp/data.npy")
    inactives_file = data_dir / Path("inactives/inactives/ecfp/data.npy")
    # output_train_path = output_dir / "train.npy"
    # output_test_path = output_dir / "test.npy"
    # print(f"{data_file}\t{output_train_path}\t{output_test_path}")
    actives_data = np.load(actives_file)
    inactives_data = np.load(inactives_file)

    actives_data = np.concatenate([actives_data, np.ones((actives_data.shape[0], 1))], axis=1)

    inactives_data = np.concatenate([inactives_data, np.zeros((inactives_data.shape[0], 1))], axis=1)

    # print(actives_data.shape, inactives_data.shape)


    data = np.concatenate([actives_data, inactives_data], axis=0)


    # print(int(actives_data.shape[0]*.2), int(actives_data.shape[0]*.8),int(actives_data.shape[0]*.2) + int(actives_data.shape[0]*.8), actives_data.shape[0])
    # print(int(inactives_data.shape[0]*.2), int(inactives_data.shape[0]*.8), int(inactives_data.shape[0]*.2) + int(inactives_data.shape[0]*.8), inactives_data.shape[0])
    # label_index = -1

    # label = 0
    # if data_file.parent == "actives":
        # label = 1


    x_train, x_test = train_test_split(data, stratify=data[:, -1], test_size=0.2)

    
    train_path = data_dir / Path("random_split_train.npy")
    test_path = data_dir / Path("random_split_test.npy")

    print(train_path, x_train.shape, test_path, x_test.shape)
    np.save(train_path, x_train)
    np.save(test_path, x_test)




if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', choices=["dude", "lit-pcba"])

    args = parser.parse_args()

    if args.dataset == "dude":
        for data_file in Path("datasets/dude/deepchem_feats").glob("**/data.npy"):
            # print(data_file)
            output_dir = data_file.parent
            random_partition_dude_data(data_file=data_file, output_dir=data_file.parent)

    elif args.dataset == "lit-pcba":

        for data_dir in Path("/g/g13/jones289/workspace/hd-cuda-master/datasets/lit_pcba/lit_pcba_full_data").glob("*"):
            partition_lit_pcba_data(data_dir)
