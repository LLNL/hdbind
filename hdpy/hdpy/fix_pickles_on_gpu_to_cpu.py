import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--random-seed")
parser.add_argument("--dataset")
args = parser.parse_args()


from pathlib import Path

result_dir = Path(f"results/{args.random_seed}/")


import pickle


def load_pickle(path):

    with open(path, "rb") as handle:
        data = pickle.load(handle)

    return data


def save_pickle(pkl_path, pkl_data):

    with open(pkl_path, "wb") as handle:
        pickle.dump(pkl_data, handle)


import torch
# import pdb
# pdb.set_trace()
for path in result_dir.glob(f"{args.dataset}*.pkl"):
    print(path)
    
    data = load_pickle(path)

    model = data[0]["model"]

    if isinstance(model, torch.nn.Module):
        for i in range(10):
            # model_i = data[i]["model"]
            data[i]["model"].am[0] = data[i]["model"].am[0].to('cpu')
            data[i]["model"].am[1] = data[i]["model"].am[1].to('cpu')
            data[i]["model"] = data["model"].to('cpu')

    # save the update result

    save_pickle(path, data)

