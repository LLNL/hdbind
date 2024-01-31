################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
from cProfile import run

# import pickle
# import time
import torch
import random
import time
random.seed(time.time())
# import ray
from tqdm import tqdm
import numpy as np
import pandas as pd
import selfies as sf

constrain_dict = sf.get_semantic_constraints()
from hdpy.data_utils import MolFormerDataset, ECFPFromSMILESDataset, SMILESDataset
import deepchem as dc

# from deepchem.molnet import load_hiv, load_tox21, load_bace_classification, load_sider
# from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# from sklearn.metrics import classification_report
# from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset
from pathlib import Path
from hdpy.ecfp.encode import ECFPEncoder
from hdpy.model import RPEncoder, run_mlp, run_hd
from hdpy.selfies_enc.encode import SELFIESHDEncoder
from hdpy.model import TokenEncoder

SCRATCH_DIR = "/p/vast1/jones289/"


def main(
    model,
    train_dataset,
    test_dataset,
):
    if config.model in ["molehd", "selfies", "ecfp", "rp"]:
        result_dict = run_hd(
            model=model,
            config=config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_trials=args.n_trials,
            random_state=args.random_state,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )
    elif config.model in ["mlp"]:
        result_dict = run_mlp(
            config=config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_trials=args.n_trials,
            random_state=args.random_state,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )
    else:
        raise NotImplementedError

    return result_dict


def driver():
    train_dataset, test_dataset = None, None
    # todo: ngram order and tokenizer only apply to some models, don't need to have in the exp_name
    if config.model == "molehd":
        model = TokenEncoder(D=config.D, num_classes=2)
        # will update item_mem after processing input data

    elif config.model == "selfies":
        model = SELFIESHDEncoder(D=config.D)

    elif config.model == "ecfp":
        model = ECFPEncoder(D=config.D)

    elif config.model == "rp":
        # assert config.ecfp_length is not None
        assert config.D is not None
        model = RPEncoder(input_size=config.input_size, D=config.D, num_classes=2)

    else:
        # if using sklearn or pytorch non-hd model
        model = None

    if config.model in ["smiles-pe", "selfies", "ecfp", "rp"]:
        # transfer the model to GPU memory
        model = model.to(device).float()

        print("model is on the gpu")

    output_result_dir = Path(f"results/{args.random_state}")
    if not output_result_dir.exists():
        output_result_dir.mkdir(parents=True, exist_ok=True)

    print(config)

    result_dict = None
    roc_values = (
        []
    )  # some datasets contain multiple targets, store these values then print at end

    smiles_featurizer = dc.feat.DummyFeaturizer()

    # if args.split_type == "random":
    # lit_pcba_data_p = Path("/p/vast1/jones289/lit_pcba/lit_pcba_full_data/")
    lit_pcba_ave_p = Path("/p/vast1/jones289/lit_pcba/AVE_unbiased")

    target_list = list(lit_pcba_ave_p.glob("*/"))

    random.shuffle(target_list)
    for target_path in tqdm(target_list):
        target_name = target_path.name
        output_file = Path(
            f"{output_result_dir}/{exp_name}.{args.dataset}-{target_path.name}-{args.split_type}.{args.random_state}.pkl"
        )
        if output_file.exists():
            print(f"output_file: {output_file} exists. skipping.")
            result_dict = torch.load(output_file)

        else:
            smiles_train, smiles_test, y_train, y_test = None, None, None, None

            df = pd.read_csv(
                target_path / Path("full_data.csv")
                # target_path.parent.parent
                # / target_path.parent
                # / Path(target_path.name)
                # / Path("full_smiles_with_labels.csv")
            )
            # import pdb
            # pdb.set_trace()
            # df["smiles"] = df[0]
            df["index"] = df.index

            # import pdb
            # pdb.set_trace()

            # load the smiles strings, if split type is ave then the split has already been computed, other wise load the
            # corresponding file and do the split
            if args.split_type == "random":
                # raise NotImplementedError

                # df = pd.read_csv(target_path / Path("full_smiles_with_labels.csv"))

                _, test_idxs = train_test_split(
                    list(range(len(df))),
                    stratify=df["label"],
                    random_state=args.random_state,
                )

                df["split"] = ["train"] * len(df)
                df.loc[test_idxs, "split"] = "test"

                train_df = df[df["split"] == "train"]
                test_df = df[df["split"] == "test"]


                smiles_train = (df["0"][df[df["split"] == "train"]["index"]]).values
                smiles_test = (df["0"][df[df["split"] == "test"]["index"]]).values

                y_train = (df["label"][df[df["split"] == "train"]["index"]]).values
                y_test = (df["label"][df[df["split"] == "test"]["index"]]).values

            else:
                train_df = pd.read_csv(
                    # lit_pcba_ave_p / Path(f"{target_name}/smiles_train.csv"),
                    lit_pcba_ave_p / Path(f"{target_name}/train_data.csv"),
                )
                smiles_train = train_df['0'].values

                test_df = pd.read_csv(
                    # lit_pcba_ave_p / Path(f"{target_name}/smiles_test.csv"), header=None
                    lit_pcba_ave_p / Path(f"{target_name}/test_data.csv")
                )

                #todo: there may be an issue with how smiles_test is being saved for molformer
                smiles_test = test_df['0'].values

                # train_df = pd.merge(df, pd.DataFrame({"smiles": smiles_train}))
                y_train = train_df["label"].values
                # test_df = pd.merge(df, pd.DataFrame({"smiles": smiles_test}))
                y_test = test_df["label"].values
                # import pdb
                # pdb.set_trace()

            # import pdb
            # pdb.set_trace()
            if config.embedding == "ecfp":

                train_dataset = ECFPFromSMILESDataset(
                    smiles=smiles_train,
                    labels=y_train,
                    ecfp_length=config.ecfp_length,
                    ecfp_radius=config.ecfp_radius,
                )

                test_dataset = ECFPFromSMILESDataset(
                    smiles=smiles_test,
                    labels=y_test,
                    ecfp_length=config.ecfp_length,
                    ecfp_radius=config.ecfp_radius,
                )

            elif config.embedding in ["atomwise", "ngram", "selfies", "bpe"]:
                # its assumed in this case you are using an HD model, this could change..
                train_dataset = SMILESDataset(
                    smiles=smiles_train,
                    labels=y_train,
                    D=config.D,
                    tokenizer=config.embedding,
                    ngram_order=config.ngram_order,
                    num_workers=16,
                    device=device,
                )
                # use the item_memory generated by the train_dataset as a seed for the test, then update both?
                test_dataset = SMILESDataset(
                    smiles=smiles_test,
                    labels=y_test,
                    D=config.D,
                    tokenizer=config.embedding,
                    ngram_order=config.ngram_order,
                    item_mem=train_dataset.item_mem,
                    num_workers=1,
                    device=device,
                )

                train_dataset.item_mem = test_dataset.item_mem
                model.item_mem = train_dataset.item_mem

            elif config.embedding in ["molformer", "molformer-ecfp-combo"]:

                if args.split_type == "random":
                    # import `pdb
                    # pdb.set_trace()

                    full_molformer_path = lit_pcba_ave_p / Path(target_name) / Path(
                    "molformer_embedding_N-Step-Checkpoint_3_30000_full.npy"
                    )
                    if config.embedding == "molformer-ecfp-combo":
                        print("loading combo model")
                        full_molformer_path = lit_pcba_ave_p / Path(target_name) / Path(
                        f"molformer_embedding_ecfp_{config.ecfp_length}_{config.ecfp_radius}_N-Step-Checkpoint_3_30000_full.npy"
                        )


                    data = np.load(full_molformer_path)

                    # data = load_full_data_molformer_embeddings() # this file should align with the full_data.csv
                    train_data = data[train_df["index"].values, :]
                    test_data = data[test_df["index"].values, :]

                    # raise NotImplementedError
                    # x_train = norm.fit_transform(train_data[:, :-1]) # according to sklearn docs, calling fit does nothing and this is the preferred way of doing things...
                    x_train = train_data[:, :-1] # according to sklearn docs, calling fit does nothing and this is the preferred way of doing things...
                    y_train = train_data[:, -1]

                    # x_test = norm.fit_transform(test_data[:, :-1])
                    x_test = test_data[:, :-1]
                    y_test = test_data[:, -1]

                    train_dataset = TensorDataset(
                        torch.from_numpy(x_train).float(),
                        torch.from_numpy(y_train).int(),
                    )
                    test_dataset = TensorDataset(
                        torch.from_numpy(x_test).float(),
                        torch.from_numpy(y_test).int(),
                    )

                else:
                    train_molformer_path = lit_pcba_ave_p / Path(target_name) / Path(
                    "molformer_embedding_N-Step-Checkpoint_3_30000_train.npy"
                    )

                    test_molformer_path = lit_pcba_ave_p / Path(target_name) / Path(
                    "molformer_embedding_N-Step-Checkpoint_3_30000_test.npy"
                    )

                    if config.embedding == "molformer-ecfp-combo":
                        print("loading combo model")
                        train_molformer_path = lit_pcba_ave_p / Path(target_name) / Path(
                        f"molformer_embedding_ecfp_{config.ecfp_length}_{config.ecfp_radius}_N-Step-Checkpoint_3_30000_train.npy"
                        )
                        test_molformer_path = lit_pcba_ave_p / Path(target_name) / Path(
                        f"molformer_embedding_ecfp_{config.ecfp_length}_{config.ecfp_radius}_N-Step-Checkpoint_3_30000_test.npy"
                        )


                    train_data = np.load(train_molformer_path)
                    test_data = np.load(test_molformer_path)

                    # train_idxs = train_df["index"].values
                    # test_idxs = test_df["index"].values

                    from sklearn.preprocessing import Normalizer

                    norm = Normalizer(norm="l2")

                    # x_train = norm.fit_transform(train_data[:, :-1]) # according to sklearn docs, calling fit does nothing and this is the preferred way of doing things...
                    x_train = train_data[:, :-1] # according to sklearn docs, calling fit does nothing and this is the preferred way of doing things...
                    y_train = train_data[:, -1]

                    # x_test = norm.fit_transform(test_data[:, :-1])
                    x_test = test_data[:, :-1]
                    y_test = test_data[:, -1]

                    train_dataset = TensorDataset(
                        torch.from_numpy(x_train).float(),
                        torch.from_numpy(y_train).int(),
                    )
                    test_dataset = TensorDataset(
                        torch.from_numpy(x_test).float(),
                        torch.from_numpy(y_test).int(),
                    )
            else:
                raise NotImplementedError

            result_dict = main(
                model=model, train_dataset=train_dataset, test_dataset=test_dataset
            )

            # import pdb
            # pdb.set_trace()
            result_dict["smiles_train"] = smiles_train
            result_dict["smiles_test"] = smiles_test
            result_dict["y_train"] = y_train
            result_dict["y_test"] = y_test

            result_dict["args"] = config
            torch.save(result_dict, output_file)
            print(f"done. output file: {output_file}")

        roc_values.append(
            np.mean([value["roc-auc"] for value in result_dict["trials"].values()])
        )

    print(f"Average ROC-AUC is {np.mean(roc_values)} +/- ({np.std(roc_values)})")


if __name__ == "__main__":
    import argparser

    # args contains things that are unique to a specific run
    args = argparser.parse_args()
    # config contains general information about the model/data processing
    config = argparser.get_config(args)

    if config.device == "cpu":
        device = "cpu"
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"using device {device}")

    exp_name = f"{Path(args.config).stem}"

    driver()
