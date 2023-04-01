
import pickle
import time
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from hdpy.metrics import compute_enrichment_factor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tqdm.write(f"using device {device}")


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=[
        "bbbp",
        "sider",
        "clintox",
        "dude",
        "lit-pcba",
        "lit-pcba-ave",
        "dockstring",
    ],
    required=True,
)
parser.add_argument("--split-type", choices=["random", "scaffold"], required=True)
parser.add_argument("--model", choices=["smiles-pe", "selfies", "ecfp", "rp", "rf", "mlp"])
parser.add_argument("--tokenizer", choices=["atomwise", "ngram", "bpe", "selfies-charwise", "None"], default="None") #TODO: have None as an option since e.g. sklearn models don't use this option
parser.add_argument("--ngram-order", type=int, default=0, help="specify the ngram order, 1-unigram, 2-bigram, so on. 0 is default to trigger an error in case ngram is specified as the tokenizer, we don't use this arg for atomwise or bpe")
parser.add_argument("--D", type=int, help="size of encoding dimension", default=10000)
# parser.add_argument(
    # "--input-feat-size", type=int, help="size of input feature dim. ", default=1024
# )
parser.add_argument(
    "--n-trials", type=int, default=1, help="number of trials to perform"
)
# parser.add_argument("--hd-retrain-epochs", type=int, default=1)
parser.add_argument("--random-state", type=int, default=0)
parser.add_argument("--initial-p", type=float, default=0.1)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--plist", nargs="+",type=float, default=[.1, .01])
args = parser.parse_args()

# seed the RNGs
import random

def seed_rngs(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test(model, hv_test, y_test):

    test_start = time.time()
    pred_list = model.predict(hv_test)
    test_time = time.time() - test_start

    conf_test_start = time.time()
    eta_list = model.compute_confidence(hv_test)
    conf_test_time = time.time() - conf_test_start


    return {"y_pred": pred_list, "y_true": y_test, "eta": eta_list, "test_time": test_time, "conf_test_time": conf_test_time}



# def main(smiles, labels, train_idxs, test_idxs):
def main(x_train, y_train, x_test, y_test, smiles_train=None, smiles_test=None):
    
    # run inference using HD model

    # get top 1% of predicted molecules 

    # get top 10% of predicted molecules


    # run inference on top 1% of predicted molecules using MLP

    # run inference on top 10% of predicted molecules using MLP


    # tqdm.write enrichment for top 1% using only HD and top 1% using HD->MLP

    # tqdm.write enrichment for top 10% using only HD and top 10% using HD-> MLP

    pass

if __name__ == "__main__":

    """
    each problem (bbbp, sider, clintox) have n binary tasks..we'll form a separate AM for each
    """


    output_dict = {"target": [], "enrich": [], "p": [], "actives_database": [], "hdc_actives_sampled": [], "trial": []}


    output_result_dir = Path(f"results/{args.random_state}")
    if not output_result_dir.exists():
        output_result_dir.mkdir(parents=True, exist_ok=True)

    print(args)

    hd_cache_dir = f"/p/lustre2/jones289/hd_cache/{args.random_state}/{args.model}/{args.dataset}/{args.split_type}"
    p_list = args.plist


    if args.dataset == "dude":
        raise NotImplementedError


    elif args.dataset == "lit-pcba":

        lit_pcba_data_p = Path(
            "/usr/WS1/jones289/hd-cuda-master/datasets/lit_pcba/lit_pcba_full_data/"
        )

        enrich_list = []

        for lit_pcba_path in lit_pcba_data_p.glob("*"):

            target_name = lit_pcba_path.name
            result_file = Path(f"{output_result_dir}/{args.dataset.replace('-', '_')}.{target_name}.{args.model}.{args.tokenizer}.{args.ngram_order}.pkl")


            target_name = lit_pcba_path.name

            target_test_hv_path = f"{hd_cache_dir}/{target_name}/test_dataset_hv.pth"


            result_file = Path(f"{output_result_dir}/{args.dataset.replace('-', '_')}.{target_name}.{args.model}.{args.tokenizer}.{args.ngram_order}.pkl")
            mlp_result_file = Path(f"{output_result_dir}/{args.dataset.replace('-', '_')}.{target_name}.mlp.None.{args.ngram_order}.pkl")



            for trial in range(10):
                print(trial, result_file, mlp_result_file)

                with open(mlp_result_file, "rb") as handle:
                    mlp_result_dict = pickle.load(handle)
                    mlp_model = mlp_result_dict[trial]["model"]


                if result_file.exists():
                    model = None 
                    with open(result_file, "rb") as handle:
                        result_dict = pickle.load(handle)
                        model = result_dict[trial]["model"]
                        model = model.to(device)
                        y_test = result_dict['y_test']

                    hv_test = torch.load(target_test_hv_path).to(device)
                    
                    conf_scores = model.compute_confidence(hv_test.cpu())

                    hd_enrich = compute_enrichment_factor(scores=conf_scores, labels=y_test, n_percent=args.initial_p)

                    values, idxs = torch.sort(conf_scores.squeeze().cpu(), descending=True)

                    sample_n = int(np.ceil(args.initial_p * y_test.shape[0]))

                    hd_actives = sum(y_test[idxs[:sample_n]])


                    actives_database = sum(y_test)

                    for p in p_list:

                        # get the indexes of the top initial-p% of compounds ranked by HDC
                        samp_idxs = (idxs[:sample_n]).numpy()


                        # take result of filtering from HDC
                        x_test_samp = result_dict["x_test"][samp_idxs]
                        y_test_samp = y_test[samp_idxs]

                        sorted_mlp_scores = sorted(zip(mlp_model.predict_proba(x_test_samp)[:, 1], y_test_samp), reverse=True)

                        # now we're going to take the top p% from the top 1% of compounds ranked by HDC, p*1% of library
                        samp_n = int(np.ceil(p * sample_n))

                        top_n_mlp_sorted_scores = sorted_mlp_scores[:samp_n]

                        actives_sampled = sum([y for x,y in top_n_mlp_sorted_scores])

                        enrich = (actives_sampled/actives_database) * (y_test.shape[0]/samp_n)

                        tqdm.write(f"target: {target_name}, data_size: {hv_test.shape[0]}, initial_p: {args.initial_p}, samp_n_10: {sample_n}, x_test_samp: {x_test_samp.shape},p: {p}, samp_n: {samp_n}, hd_enrich: {hd_enrich},mlp_enrich_{int(p*10)}: {enrich}, hdc_actives_sampled: {hd_actives} ,mlp_actives_sampled: {actives_sampled}, actives_database: {actives_database}")


                        output_dict["target"].append(target_name)
                        output_dict["enrich"].append(enrich)
                        output_dict["p"].append(p)
                        output_dict["actives_database"].append(actives_database)
                        output_dict["hdc_actives_sampled"].append(hd_actives)
                        output_dict["trial"].append(trial)
                    
                    
                    
                    df = pd.DataFrame(output_dict)

                    enrich_list.append(df)
                else:
                    tqdm.write(f"result file: {result_file} for input file {lit_pcba_path} doesn't exist. moving on to next target..")


    full_df = pd.concat(enrich_list)
    full_df.to_csv(f"{args.random_state}_{args.dataset}_{args.initial_p}_multistep_enrich.csv")


