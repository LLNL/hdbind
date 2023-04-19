import argparse
import torch
parser = argparse.ArgumentParser()

parser.add_argument("--random-seed")
parser.add_argument("--dataset")
parser.add_argument("--model")
args = parser.parse_args()


from pathlib import Path

result_dir = Path(f"results/{args.random_seed}/")


import pickle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm 
tqdm.write(f"using device {device}")

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







hd_cache_dir = f"/p/lustre2/jones289/hd_cache/{args.random_seed}/{args.model}/{args.dataset}/random"




path_list = list(result_dir.glob(f"{args.dataset.replace('-', '_')}*.pkl"))

for path in tqdm(path_list, total=len(path_list)):
    print(path)
    
    data = load_pickle(path)

    model = data[0]["model"]

    if isinstance(model, torch.nn.Module):
        # hv_test =  

        # if args.dataset == "lit-pcba":
        target_name = path.name.split('.')[1]

        # import pdb 
        # pdb.set_trace()
        target_test_hv_path = f"{hd_cache_dir}/{target_name}/test_dataset_hv.pth"



        hv_test = torch.load(target_test_hv_path).to(device)



        for i in range(10):
            model_i = data[i]["model"].to(device)
            model_i.am[0] = model_i.am[0].to(device)
            model_i.am[1] = model_i.am[1].to(device)

            # import pdb 
            # pdb.set_trace()
            conf_scores = model_i.compute_confidence(hv_test)


            data[i]["model"].am[0] = data[i]["model"].am[0].to('cpu')
            data[i]["model"].am[1] = data[i]["model"].am[1].to('cpu')
            # data[i]["model"] = data[i]["model"].to('cpu')




            data[i]["eta"] = conf_scores.to('cpu').numpy()

    # save the update result

    save_pickle(path, data)

