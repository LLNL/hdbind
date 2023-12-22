################################################################################
# Copyright (c) 2021-2023, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
from cProfile import run
import torch
import numpy as np
import pandas as pd
import random
import selfies as sf
constrain_dict = sf.get_semantic_constraints()
from hdpy.data_utils import MolFormerDataset, ECFPFromSMILESDataset, SMILESDataset
import deepchem as dc
from deepchem.molnet import load_hiv, load_tox21, load_bace_classification, load_sider
from pathlib import Path
from hdpy.ecfp.encode import ECFPEncoder
from hdpy.model import RPEncoder, run_mlp, run_hd
from hdpy.selfies.encode import SELFIESHDEncoder
from hdpy.utils import compute_splits
from hdpy.model import TokenEncoder
from deepchem.molnet import load_bace_classification
from torch.utils.data import TensorDataset
# SCRATCH_DIR = "/p/lustre2/jones289/"
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
        result_dict = run_mlp(config=config, batch_size=args.batch_size,
                              num_workers=args.num_workers, n_trials=args.n_trials,
                              random_state=args.random_state,
                              train_dataset=train_dataset, test_dataset=test_dataset)
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
    roc_values = [] # some datasets contain multiple targets, store these values then print at end


    smiles_featurizer = dc.feat.DummyFeaturizer()
    
    if args.dataset == "bbbp":


        smiles_col = "rdkitSmiles"
        label_col = "p_np"
        target_list = [label_col]
        df = pd.read_csv(
            "/g/g13/jones289/workspace/hd-cuda-master/datasets/moleculenet/BBBPMoleculesnetMOE3D_rdkitSmilesInchi.csv"
        )

        split_path = Path(
            f"{output_result_dir}/{args.dataset}.{args.split_type}.{args.random_state}.train_test_split.csv"
        )

        split_df = compute_splits(
            split_path=split_path,
            random_state=args.random_state,
            split_type=args.split_type,
            df=df,
            smiles_col=smiles_col,
            label_col=label_col
        )

        smiles_train = (split_df[split_df.loc[:, "split"] == "train"][smiles_col]).values
        smiles_test = (split_df[split_df.loc[:, "split"] == "test"][smiles_col]).values

        y_train = (split_df[split_df.loc[:, "split"] == "train"][label_col]).values.reshape(-1,len(target_list)) 
        y_test = (split_df[split_df.loc[:, "split"] == "test"][label_col]).values.reshape(-1,len(target_list))

    elif args.dataset == "sider":
        sider_dataset = load_sider()

        target_list = sider_dataset[0]


        smiles_train = sider_dataset[1][0].ids
        smiles_test = sider_dataset[1][1].ids

        y_train = sider_dataset[1][0].y
        y_test = sider_dataset[1][1].y
        
    elif args.dataset == "clintox":

        smiles_col="smiles"
        label_col="CT_TOX"
        target_list = [label_col]
        df = pd.read_csv(
            "/g/g13/jones289/workspace/hd-cuda-master/datasets/moleculenet/clintox-cleaned.csv"
        )

        # some of the smiles (6) for clintox dataset are not able to be converted to fingerprints, so skip them for all cases

        split_path = Path(
            f"{output_result_dir}/{args.dataset}.{args.split_type}.{args.random_state}.train_test_split.csv"
        )

        split_df = compute_splits(
            split_path=split_path,
            random_state=args.random_state,
            split_type=args.split_type,
            df=df,
            smiles_col=smiles_col,
            label_col=label_col,
        )
        
        smiles_train = (split_df[split_df.loc[:, "split"] == "train"][smiles_col]).values
        smiles_test = (split_df[split_df.loc[:, "split"] == "test"][smiles_col]).values

        y_train = (split_df[split_df.loc[:, "split"] == "train"][label_col]).values.reshape(-1,len(target_list)) 
        y_test = (split_df[split_df.loc[:, "split"] == "test"][label_col]).values.reshape(-1,len(target_list))

    elif args.dataset == "bace":

        dataset = load_bace_classification(splitter="scaffold", featurizer=smiles_featurizer)
        target_list = dataset[0]
        train_dataset = dataset[1][0]
        test_dataset = dataset[1][1]

        smiles_train = train_dataset.X
        y_train = train_dataset.y

        smiles_test = test_dataset.X
        y_test = test_dataset.y

    elif args.dataset == "tox21":

        dataset = load_tox21(splitter="scaffold", featurizer=smiles_featurizer)
        
        target_list = dataset[0]
        train_dataset = dataset[1][0]
        test_dataset = dataset[1][1]

        smiles_train = train_dataset.X
        y_train = train_dataset.y

        smiles_test = test_dataset.X
        y_test = test_dataset.y

    elif args.dataset == "hiv":

        dataset = load_hiv(splitter="scaffold", featurizer=smiles_featurizer)
        target_list = dataset[0]

        # use something besides train_dataset/test_dataset?
        train_dataset = dataset[1][0]
        test_dataset = dataset[1][1]

        smiles_train = train_dataset.X
        y_train = train_dataset.y

        smiles_test = test_dataset.X
        y_test = test_dataset.y


    random.shuffle(target_list)
    for target_idx, target_name in enumerate(target_list):

        if config.embedding == "ecfp":
            train_dataset = ECFPFromSMILESDataset(smiles=smiles_train, 
                                        labels=y_train[:, target_idx], 
                                        ecfp_length=config.ecfp_length,
                                        ecfp_radius=config.ecfp_radius)
            
            test_dataset = ECFPFromSMILESDataset(smiles=smiles_test,
                                    labels=y_test[:, target_idx],
                                    ecfp_length=config.ecfp_length,
                                    ecfp_radius=config.ecfp_radius)

        elif config.embedding in ["atomwise", "ngram", "selfies", "bpe"]:
            # its assumed in this case you are using an HD model, this could change..
            train_dataset = SMILESDataset(
                smiles=smiles_train,
                labels=y_train[:, target_idx],
                D=config.D,
                tokenizer=config.embedding,
                ngram_order=config.ngram_order,
                num_workers=16,
                device=device,
            )
            # use the item_memory generated by the train_dataset as a seed for the test, then update both?
            test_dataset = SMILESDataset(
                smiles=smiles_test,
                labels=y_test[:, target_idx],
                D=config.D,
                tokenizer=config.embedding,
                ngram_order=config.ngram_order,
                item_mem=train_dataset.item_mem,
                num_workers=1,
                device=device,
            )

            train_dataset.item_mem = test_dataset.item_mem
            model.item_mem = train_dataset.item_mem

        elif config.embedding == "molformer":

            train_data = np.load(f"{SCRATCH_DIR}/molformer_embeddings/molnet/{args.dataset}/train_N-Step-Checkpoint_3_30000.npy")
            test_data = np.load(f"{SCRATCH_DIR}/molformer_embeddings/molnet/{args.dataset}/test_N-Step-Checkpoint_3_30000.npy")


            train_dataset = TensorDataset(torch.from_numpy(train_data[:, :768]).float(), 
                                          torch.from_numpy(train_data[:, (768+target_idx)]).float())
            test_dataset = TensorDataset(torch.from_numpy(test_data[:, :768]).float(), 
                                         torch.from_numpy(test_data[:, (768+target_idx)]).float())

        else:
            raise NotImplementedError



        # todo: add target list or target_idx to output_file? this is already the format for dude/lit-pcba/clintox so just extend trivially
        output_file = Path(
            f"{output_result_dir}/{exp_name}.{args.dataset}-{target_name.replace(' ','_')}-{args.split_type}.{args.random_state}.pkl"
        )
        if output_file.exists():
            print(f"output_file: {output_file} exists. skipping.")
            result_dict = torch.load(output_file)

        else:
            result_dict = main(
                    model=model, train_dataset=train_dataset, test_dataset=test_dataset
                )

            result_dict["smiles_train"] = smiles_train
            result_dict["smiles_test"] = smiles_test
            result_dict["y_train"] = y_train[:, target_idx]
            result_dict["y_test"] = y_test[:, target_idx]

            result_dict["args"] = config
            torch.save(result_dict, output_file)
            print(f"done. output file: {output_file}")

        roc_values.append(np.mean([value["roc-auc"] for value in result_dict["trials"].values()]))

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
