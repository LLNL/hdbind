
import pickle
import time
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import selfies as sf
constrain_dict = sf.get_semantic_constraints()
import torch.multiprocessing as mp
from pathlib import Path
from hdpy.metrics import compute_enrichment_factor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")


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
parser.add_argument(
    "--input-feat-size", type=int, help="size of input feature dim. ", default=1024
)
parser.add_argument(
    "--n-trials", type=int, default=1, help="number of trials to perform"
)
parser.add_argument("--hd-retrain-epochs", type=int, default=1)
parser.add_argument("--random-state", type=int, default=0)
parser.add_argument("--dry-run", action="store_true")
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


    # print enrichment for top 1% using only HD and top 1% using HD->MLP

    # print enrichment for top 10% using only HD and top 10% using HD-> MLP

    pass

if __name__ == "__main__":

    """
    each problem (bbbp, sider, clintox) have n binary tasks..we'll form a separate AM for each
    """






    # if args.model in ["smiles-pe", "selfies", "ecfp", "rp"]:
        # transfer the model to GPU memory
        # hd_model = hd_model.to(device).float()



    output_result_dir = Path(f"results/{args.random_state}")
    if not output_result_dir.exists():
        output_result_dir.mkdir(parents=True, exist_ok=True)

    print(args)


    hd_cache_dir = f"/p/lustre2/jones289/hd_cache/{args.random_state}/{args.model}/{args.dataset}/{args.split_type}"


    if args.dataset == "dude":

        if args.split_type.lower() != "random": #i.e. scaffold
            raise NotImplementedError(f"DUD-E does not support {args.split_type}")

        dude_data_p = Path("/usr/workspace/atom/gbsa_modeling/dude_smiles/")
        dude_path_list = list(dude_data_p.glob("*_gbsa_smiles.csv"))
        for dude_smiles_path in tqdm(dude_path_list):

            target_name = dude_smiles_path.name.split("_")[0]

            target_test_hv_path = f"{hd_cache_dir}/{target_name}/test_dataset_hv.pth"


            result_file = Path(f"{output_result_dir}/{args.dataset}.{args.split_type}.{target_name}.{args.model}.{args.tokenizer}.{args.ngram_order}.pkl")
            mlp_result_file = Path(f"{output_result_dir}/{args.dataset}.{args.split_type}.{target_name}.mlp.None.{args.ngram_order}.pkl")


            with open(mlp_result_file, "rb") as handle:
                mlp_result_dict = pickle.load(handle)
                mlp_model = mlp_result_dict[0]["model"]


            if result_file.exists():
                model = None 
                with open(result_file, "rb") as handle:
                    result_dict = pickle.load(handle)
                    model = result_dict[0]["model"]
                    model = model.to(device)
                    y_test = result_dict['y_test']

                hv_test = torch.load(target_test_hv_path).to(device)
                

                # model.predict(hv_test)
                # import pdb 
                # pdb.set_trace()
                conf_scores = model.compute_confidence(hv_test)

                enrich_1_hd  = compute_enrichment_factor(scores=conf_scores, labels=y_test, n_percent=.01)
                enrich_10_hd = compute_enrichment_factor(scores=conf_scores, labels=y_test, n_percent=.1)

                values, idxs = torch.sort(conf_scores.squeeze().cpu(), descending=True)


                sample_n_1 = int(np.ceil(.01 * y_test.shape[0]))
                sample_n_10 = int(np.ceil(.1 * y_test.shape[0]))

                hd_actives_1 = sum(y_test[idxs[:sample_n_1]])
                hd_actives_10 = sum(y_test[idxs[:sample_n_10]])



                actives_database = sum(y_test)

                for p in [.01, .1]:

                    # get the indexes of the top 1% of compounds ranked by HDC
                    samp_idxs = (idxs[:sample_n_1]).numpy()


                    # take result of filtering from HDC
                    x_test_samp = result_dict["x_test"][samp_idxs]
                    y_test_samp = y_test[samp_idxs]

                    sorted_mlp_scores = sorted(zip(mlp_model.predict_proba(x_test_samp)[:, 1], y_test_samp))

                    # now we're going to take the top p% from the top 1% of compounds ranked by HDC, p*1% of library
                    samp_n = int(np.ceil(p * sample_n_1))

                    top_n_mlp_sorted_scores = sorted_mlp_scores[:samp_n]

                    actives_sampled = sum([y for x,y in top_n_mlp_sorted_scores])

                    enrich = (actives_sampled/actives_database) * (y_test.shape[0]/samp_n)

                    print(f"data_size: {hv_test.shape[0]}, samp_n_1: {sample_n_1}, x_test_samp: {x_test_samp.shape},p: {p}, samp_n: {samp_n}, hdc_enrich_1: {enrich_1_hd}, mlp_enrich_1_{p}: {enrich}, hdc_actives_sampled: {hd_actives_1} ,mlp_actives_sampled: {actives_sampled}, actives_database: {actives_database}")

            else:
                print(f"result file: {result_file} for input file {dude_smiles_path} already exists. moving on to next target..")

    '''
    elif args.dataset == "lit-pcba":

        lit_pcba_data_p = Path(
            "/usr/WS1/jones289/hd-cuda-master/datasets/lit_pcba/lit_pcba_full_data/"
        )
        for lit_pcba_path in lit_pcba_data_p.glob("*"):

            target_name = lit_pcba_path.name
            output_file = Path(f"{output_result_dir}/{args.dataset.replace('-', '_')}.{target_name}.{args.model}.{args.tokenizer}.{args.ngram_order}.pkl")

            if output_file.exists():
                # don't recompute if it's already been calculated
                print(f"output file: {output_file} for input file {lit_pcba_path} already exists. moving on to next target..")
                pass  

            else:
                pass
    '''