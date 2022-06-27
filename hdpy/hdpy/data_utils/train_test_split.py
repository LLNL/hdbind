import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path 
import sys 


def partition_data(data_file, output_dir):

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


if __name__ == "__main__":

    for data_file in Path("datasets/dude/deepchem_feats").glob("**/data.npy"):
        # print(data_file)
        output_dir = data_file.parent
        partition_data(data_file=data_file, output_dir=data_file.parent)
