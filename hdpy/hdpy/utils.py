import time
from tqdm import tqdm
import numpy as np
# from hdpy.utils import convert_pdbbind_affinity_to_class_label
from torch.utils.data import Dataset


class timing_part:
    def __init__(self, TAG, verbose=False):
        self.TAG = str(TAG)
        self.total_time = 0
        self.start_time = 0
        self.verbose = verbose

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        exit_time = time.time()
        self.total_time = exit_time - self.start_time
        if self.verbose:
            tqdm.write(f"{self.TAG}\t{self.total_time}")




'''
def map_features_by_class(data, class_list, col_index=0, dataset=None):
'''
'''
        Takes a data (iterable?), list of acceptable classes, column index, and task_type for 
        the label as input and outputs a dictionary with a list of data elements that indexed
        by the unique label values.
'''
'''
    assert dataset is not None
    class_dict = {}

    for element in data:

        # will make the assumption that the provided label is some binary format (bind/no-bind, strong-bind/weak-bind, etc.) otherwise 
        # it is necessary to make an explicit case that handles the conversion of that measure to a binary format 
        element_class = None
        if dataset == "pdbbind":
            element_target = int(element[col_index]) # expecting ints, make sure this is converted to int for use as dict key
            element_class = convert_pdbbind_affinity_to_class_label(element_target)

        else:
            element_class = int(element[col_index])

        if element_class in class_list:

            # if we haven't added the class to the class_dict yet then do so
            if element_class not in class_dict.keys():
                class_dict[element_class] = []

            # append the data to the element class data list
            class_dict[element_class].append(element)

    return class_dict
'''

def load_features(path:str, dataset:str):

    features, labels = None, None
    data = np.load(path)

    if dataset == "pdbbind":
        # map the experimental -logki/kd value to a discrete category
        features = data[:, :-1]
        labels = data[:, -1]
        labels = np.asarray([convert_pdbbind_affinity_to_class_label(x) for x in labels])


        binary_label_mask = labels != 2

        return features[binary_label_mask], labels[binary_label_mask]

    elif dataset == "dude":

        features = data[:, :-1]
        labels = data[:, -1]
              
        labels = 1- labels
    else:
        raise NotImplementedError("specify a valid supported dataset type")


    # import ipdb 
    # ipdb.set_trace()
    return features, labels


def binarize(x):
    return torch.where(x>0, 1.0, -1.0)


class CustomDataset(Dataset):
    def __init__(self, features, labels):

        self.features = features 
        self.labels = labels 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]