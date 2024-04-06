################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
from cProfile import run
import torch
import numpy as np
import pickle
import random
import deepchem as dc
from deepchem.molnet import load_hiv, load_tox21, load_bace_classification, load_sider
from pathlib import Path
from sklearn.preprocessing import normalize
from hdpy.data_utils import ECFPFromSMILESDataset, SMILESDataset
from hdpy.ecfp.encode import ECFPEncoder
from hdpy.model import RPEncoder, TokenEncoder, run_mlp, run_hd, get_model
from hdpy.selfies_enc.encode import SELFIESHDEncoder
from torch.utils.data import TensorDataset

SCRATCH_DIR = "/p/vast1/jones289"

def main(args, config,
    model,
    train_dataset,
    test_dataset,
):
    if config.model in ["molehd", "selfies", "ecfp", "rp"]:
        result_dict = run_hd(
            model=model,
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_trials=args.n_trials,
            random_state=args.random_state,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )
    elif config.model in ["mlp", "mlp-small"]:
        result_dict = run_mlp(config=config, batch_size=args.batch_size, epochs=args.epochs,
                              num_workers=args.num_workers, n_trials=args.n_trials,
                              random_state=args.random_state,
                              train_dataset=train_dataset, test_dataset=test_dataset)
    else:
        raise NotImplementedError

    return result_dict


def driver():
    train_dataset, test_dataset = None, None
    # todo: ngram order and tokenizer only apply to some models, don't need to have in the exp_name

    model = get_model(config)

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
    std_values = []

    smiles_featurizer = dc.feat.DummyFeaturizer()
    target_list, smiles_train, smiles_test, y_train, y_test = [], None, None, None, None 
    if args.dataset == "bbbp":

        # the version of deepchem I'm using has issues with this function so I'm running it elsewhere first then running in the common env
        cache_path = Path("bbbp_scaffold_dataset.pkl")
        dataset = None

        if not cache_path.exists():
            # default splitter is scaffold
            dataset = dc.molnet.load_bbbp()


            target_list = dataset[0]
            smiles_train = dataset[1][0].ids
            smiles_test = dataset[1][1].ids
            y_train = dataset[1][0].y.reshape(-1,1)
            y_test = dataset[1][1].y.reshape(-1,1)
            with open(cache_path, "wb") as handle:

                pickle.dump((target_list, 
                            smiles_train,
                            smiles_test,
                            y_train,
                            y_test), handle)

        else:
            with open(cache_path, "rb") as handle:
                data = pickle.load(handle)
            target_list, smiles_train, smiles_test, y_train, y_test = data


    elif args.dataset == "sider":
        sider_dataset = load_sider()

        target_list = sider_dataset[0]


        smiles_train = sider_dataset[1][0].ids
        smiles_test = sider_dataset[1][1].ids

        y_train = sider_dataset[1][0].y
        y_test = sider_dataset[1][1].y
        
    elif args.dataset == "clintox":

        # the version of deepchem I'm using has issues with this function so I'm running it elsewhere first then running in the common env
        cache_path = Path("clintox_scaffold_dataset.pkl")
        dataset = None

        if not cache_path.exists():
            # default splitter is scaffold
            dataset = dc.molnet.load_clintox(splitter="scaffold")


            target_list = dataset[0]
            smiles_train = dataset[1][0].ids
            smiles_test = dataset[1][1].ids
            y_train = dataset[1][0].y.reshape(len(smiles_train), -1)
            y_test = dataset[1][1].y.reshape(len(smiles_test), -1)
            with open(cache_path, "wb") as handle:

                pickle.dump((target_list, 
                            smiles_train,
                            smiles_test,
                            y_train,
                            y_test), handle)

        else:
            with open(cache_path, "rb") as handle:
                data = pickle.load(handle)
            target_list, smiles_train, smiles_test, y_train, y_test = data


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

        elif config.embedding == "molclr":

            # we're just using the GIN model always 

            train_data = np.load(f"{SCRATCH_DIR}/molclr_embeddings/molnet/{args.dataset}/train_{target_name}.npy")
            test_data = np.load(f"{SCRATCH_DIR}/molclr_embeddings/molnet/{args.dataset}/test_{target_name}.npy")


            train_dataset = TensorDataset(torch.from_numpy(normalize(train_data[:, :-1], norm="l2", axis=0)).float(), 
                                          torch.from_numpy(train_data[:, -1]).float())
            test_dataset = TensorDataset(torch.from_numpy(normalize(test_data[:, :-1], norm="l2", axis=0)).float(), 
                                         torch.from_numpy(test_data[:, -1]).float())


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
            result_dict = main(args=args, config=config,
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
        std_values.append(np.std([value["roc-auc"] for value in result_dict["trials"].values()]))

    print(f"Average ROC-AUC is {np.mean(roc_values)} +/- ({np.mean(std_values)})")


if __name__ == "__main__":
    import hdpy.hdc_args as hdc_args

    # args contains things that are unique to a specific run
    args = hdc_args.parse_args()

    assert args.split_type is not None and args.dataset is not None
    if args.split_type != "scaffold":
        print(f"{args.split_type} not supported for this dataset! please use scaffold for molnet")
        assert args.split_type == "scaffold"
    # config contains general information about the model/data processing
    config = hdc_args.get_config(args)

    if config.device == "cpu":
        device = "cpu"
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"using device {device}")

    exp_name = f"{Path(args.config).stem}"

    driver()
