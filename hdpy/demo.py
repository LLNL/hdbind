################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
import torch
import numpy as np

np.seterr(all="ignore")
from rdkit.Chem import DataStructs, rdmolfiles, AllChem
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from hdpy.model import RPEncoder, MLPClassifier
from hdpy.main import train_hdc_no_encode, test_hdc
import torch.multiprocessing as mp
from hdpy.metrics import validate
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from hdpy.dataset import SMILESDataset
from hdpy.model import TokenEncoder
import numpy as np
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", type=int, default=1)
parser.add_argument("--ecfp-radius", type=int, default=2)
parser.add_argument("--ecfp-length", type=int, default=1024)
parser.add_argument("--tokenizer", default="bpe")
parser.add_argument("--ngram_order", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--dimension", type=int, default=10000)
# parser.add_argument("--device", default="cpu")
parser.add_argument("--sample-frac", type=float, default=0.1)
parser.add_argument("--n-trials", type=int, default=1)
parser.add_argument("--debug", action="store_true")

import rdkit


def ecfp_job(smiles):
    try:
        mol_start = time.perf_counter()
        mol = rdmolfiles.MolFromSmiles(smiles)
        mol_end = time.perf_counter()

        if mol is not None:
            fp_start = time.perf_counter()
            fp_vec = AllChem.GetMorganFingerprintAsBitVect(
                mol, args.ecfp_radius, nBits=args.ecfp_length
            )
            fp_end = time.perf_counter()

            fp = np.unpackbits(
                np.frombuffer(DataStructs.BitVectToBinaryText(fp_vec), dtype=np.uint8),
                bitorder="little",
            )
            # print(type(fp))
            # return (fp_vec, end - start)
            # return {"ecfp": torch.from_numpy(fp_vec), "smiles_to_mol_time": mol_end - mol_start,
            # "mol_to_fp_time": fp_end - fp_start}
            return {
                "ecfp": fp,
                "smiles_to_mol_time": mol_end - mol_start,
                "mol_to_fp_time": fp_end - fp_start,
            }
            # return fp_vec
        else:
            return None

    except Exception as e:
        print(e)
        return None


def rp_ecfp_hdc(ecfp_list):
    # import pdb
    # pdb.set_trace()
    enc_start = time.perf_counter()
    enc = RPEncoder(D=args.dimension, input_size=args.ecfp_length, num_classes=2)
    enc.to("cuda")
    enc_end = time.perf_counter()

    # import pdb
    # pdb.set_trace()
    # print(f"random projection encoder created in {enc_end - enc_start} seconds on device={next(enc.rp_layer.parameters()).device}")

    start = time.perf_counter()
    ecfp_arr = np.array(ecfp_list)
    # dataset = TensorDataset(torch.Tensor(np.array(ecfp_list)).int())
    dataset = TensorDataset(torch.Tensor(ecfp_arr).int())
    end = time.perf_counter()
    # print(f"converting {len(ecfp_list)} ecfp list to pytorch TensorDataset took {end - start} seconds")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    ecfp_encode_time_list = []

    encoding_list = []
    for trial in range(args.n_trials):
        epoch_encode_time_sum = 0
        for batch in tqdm(dataloader, desc="encoding rp-ecfp.."):
            if args.debug:
                import pdb

                pdb.set_trace()
            x = batch[0].to("cuda")
            ecfp_encode_start = time.perf_counter()
            encodings = enc(x)
            ecfp_encode_end = time.perf_counter()

            if trial == 0:
                encoding_list.append(encodings.cpu())

            epoch_encode_time_sum += ecfp_encode_end - ecfp_encode_start

        ecfp_encode_time_list.append(epoch_encode_time_sum)

    # print(f"HDBind-RP-ECFP finished encoding in {(sum(ecfp_encode_time_list)/args.n_trials)/len(ecfp_list)} seconds (avg).")

    if args.debug:
        import pdb

        pdb.set_trace()
    return torch.cat(encoding_list), (sum(ecfp_encode_time_list) / args.n_trials) / len(
        ecfp_list
    )


def ecfp_timing(pool, smiles_list):
    result_list = list(
        tqdm(
            pool.imap(ecfp_job, smiles_list),
            total=len(smiles_list),
            desc="computing ECFPs",
        )
    )
    result_list = [x for x in result_list if x is not None]
    ecfp_list = [x["ecfp"] for x in result_list if x["ecfp"] is not None]
    smiles_to_mol_time_list = [
        x["smiles_to_mol_time"] for x in result_list if x["ecfp"] is not None
    ]
    ecfp_time_list = [x["mol_to_fp_time"] for x in result_list if x["ecfp"] is not None]

    return ecfp_list, ecfp_time_list, smiles_to_mol_time_list


def smiles_token_hdc(smiles_list):
    if args.tokenizer == "bpe":
        print(f"MoleHD-inspired baseline, tokenizer={args.tokenizer}")
    else:
        print(
            f"MoleHD-inspired baseline, tokenizer={args.tokenizer}-{args.ngram_order}"
        )

    dataset = SMILESDataset(
        smiles=smiles_list,
        tokenizer=args.tokenizer,
        ngram_order=args.ngram_order,
        D=args.dimension,
    )

    item_mem_time = dataset.item_mem_time

    tok_encoder_start = time.perf_counter()
    tok_encoder = TokenEncoder(
        D=args.dimension, num_classes=2, item_mem=dataset.item_mem
    )
    tok_encoder_end = time.perf_counter()
    # print(f"TokenEncoder created in {tok_encoder_end-tok_encoder_start} seconds.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda data: [x for x in data],
    )

    tok_encode_time_list = []

    if args.tokenizer == "bpe":
        desc = f"encoding MoleHD-{args.tokenizer}"
    else:
        desc = f"encoding MoleHD-{args.tokenizer}-{args.ngram_order}"

    for trial in range(args.n_trials):
        epoch_encode_time_sum = 0
        for batch in tqdm(dataloader, desc=desc + f"-trial({trial})"):
            if args.debug:
                import pdb

                pdb.set_trace()
            tok_encode_start = time.perf_counter()
            tok_encoder.encode_batch(batch)
            tok_encode_end = time.perf_counter()

            epoch_encode_time_sum += tok_encode_end - tok_encode_start

        tok_encode_time_list.append(epoch_encode_time_sum)

    if args.debug:
        import pdb

        pdb.set_trace()

    return (sum(tok_encode_time_list) / args.n_trials) / len(dataset)


def lit_pcba_main():
    pool = mp.Pool(args.num_workers)
    path = Path(
        "/g/g13/jones289/workspace/hd-cuda-master/datasets/lit_pcba/AVE_unbiased/ALDH1/"
    )

    train_path = path / Path("smiles_train.csv")
    test_path = path / Path("smiles_test.csv")

    train_npy_path = path / Path("ecfp_train.npy")
    test_npy_path = path / Path("ecfp_test.npy")

    train_df = pd.read_csv(train_path, header=None)
    train_npy = np.load(train_npy_path)
    train_df["label"] = train_npy[:, -1]
    train_df = train_df.sample(frac=args.sample_frac)
    train_npy = train_npy[train_df.index, :]
    # import pdb
    # pdb.set_trace()

    test_df = pd.read_csv(test_path, header=None)
    test_npy = np.load(test_npy_path)
    test_df["label"] = test_npy[:, -1]
    test_df = test_df.sample(frac=args.sample_frac)
    test_npy = test_npy[test_df.index, :]
    df = pd.concat([train_df, test_df], axis=0)
    # import pdb
    # pdb.set_trace()
    # sample_df = df.sample(frac=args.sample_frac)
    # target_smiles = sample_df[0].values.tolist()
    target_smiles = df[0].values.tolist()
    # print()
    # print(f"processing {path} with {df.shape[0]} SMILES strings. subsampling {sample_df.shape[0]}({args.sample_frac*100}%) of the original data.")
    print(
        f"processing {path} with {df.shape[0]} SMILES strings. subsampling {df.shape[0]}"
    )

    # MoleHD-based implementation
    molehd_mean_encode_time = smiles_token_hdc(smiles_list=target_smiles)

    ecfp_list, ecfp_timing_list, smiles_to_mol_time_list = ecfp_timing(
        pool=pool, smiles_list=target_smiles
    )

    # random projection encoding
    rp_ecfp_encodings, hdbind_rp_ecfp_mean_encode_time = rp_ecfp_hdc(
        ecfp_list=ecfp_list
    )

    print(
        f"MoleHD ({args.tokenizer}) encoded {len(df)} molecules at a rate of {molehd_mean_encode_time} seconds/molecule"
    )
    print(
        f"HDBind-RP-ECFP encoded {len(df)} molecules at a rate of {hdbind_rp_ecfp_mean_encode_time} seconds/molecule"
    )

    print(
        f"HDBind-RP-ECFP encoding speedup to MoleHD (bpe) is approximately {molehd_mean_encode_time/hdbind_rp_ecfp_mean_encode_time:0.2f}X"
    )

    # import pdb
    # pdb.set_trace()

    rp_ecfp_dataset_train = TensorDataset(
        rp_ecfp_encodings[list(range(len(train_df))), :],
        torch.from_numpy(train_df["label"].values),
    )
    rp_ecfp_dataset_test = TensorDataset(
        rp_ecfp_encodings[list(range(len(train_df), len(train_df) + len(test_df))), :],
        torch.from_numpy(test_df["label"].values),
    )

    rp_ecfp_dataloader_train = DataLoader(
        rp_ecfp_dataset_train, num_workers=args.num_workers, batch_size=args.batch_size
    )
    rp_ecfp_dataloader_test = DataLoader(
        rp_ecfp_dataset_test, num_workers=args.num_workers, batch_size=args.batch_size
    )

    ecfp_dataset_train = TensorDataset(
        torch.from_numpy(train_npy[:, :-1]), torch.from_numpy(train_df["label"].values)
    )
    ecfp_dataset_test = TensorDataset(
        torch.from_numpy(test_npy[:, :-1]), torch.from_numpy(test_df["label"].values)
    )

    ecfp_dataloader_train = DataLoader(
        ecfp_dataset_train, num_workers=args.num_workers, batch_size=args.batch_size
    )
    ecfp_dataloader_test = DataLoader(
        ecfp_dataset_test, num_workers=args.num_workers, batch_size=args.batch_size
    )
    # train HDBind-rp-ecfp
    rp_ecfp_model = RPEncoder(
        D=args.dimension, input_size=args.ecfp_length, num_classes=2
    )
    train_hdc_no_encode(
        model=rp_ecfp_model,
        train_dataloader=rp_ecfp_dataloader_train,
        device="cuda",
        num_epochs=10,
    )
    rp_test_dict = test_hdc(
        model=rp_ecfp_model, test_dataloader=ecfp_dataloader_test, device="cuda"
    )

    validate(
        labels=rp_test_dict["y_true"].numpy(),
        pred_labels=rp_test_dict["y_pred"].numpy(),
        pred_scores=rp_test_dict["eta"].numpy(),
    )

    mlp_model = MLPClassifier(
        layer_sizes=[(1024, 512), (512, 256), (256, 128), (128, 2)],
        lr=1e-3,
        activation=torch.nn.ReLU(),
        criterion=torch.nn.NLLLoss(),
        optimizer=torch.optim.SGD,
    )
    mlp_model.to("cuda")
    # TODO: train HDBind, MoleHD, and MLP

    from hdpy.main import train_mlp, val_mlp

    mlp_train_dict = train_mlp(
        model=mlp_model,
        train_dataloader=ecfp_dataloader_train,
        epochs=10,
        device="cuda",
    )
    mlp_test_dict = val_mlp(
        model=mlp_train_dict["model"],
        val_dataloader=ecfp_dataloader_test,
        device="cuda",
    )

    validate(
        labels=mlp_test_dict["y_true"],
        pred_labels=mlp_test_dict["y_pred"],
        pred_scores=mlp_test_dict["eta"][:, 1],
    )

    print(
        f"HDBind-RP-ECFP prediction speedup vs MLP: {mlp_test_dict['forward_time']/rp_test_dict['test_time']:.2f}X"
    )

    pool.close()


if __name__ == "__main__":
    args = parser.parse_args()

    print(f"creating multiprocessing pool of {args.num_workers} workers.")

    lit_pcba_main()
