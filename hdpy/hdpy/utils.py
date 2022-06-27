import time
from tqdm import tqdm

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


def convert_pdbbind_affinity_to_class_label(x, pos_thresh=8, neg_thresh=6):

    if x < neg_thresh:
        return 0
    elif x > pos_thresh:
        return 1
    else:
        return 2

