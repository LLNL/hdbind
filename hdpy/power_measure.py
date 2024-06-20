import torch
import numpy as np
from hdpy.model import get_model, train_hdc, test_hdc, encode_hdc 
from torch.utils.data import TensorDataset
import hdpy.hdc_args as hdc_args
from tqdm import tqdm 
from hdpy.utils import collate_list_fn
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import StreamingECFPDataset, StreamingComboDataset, ECFPFromSMILESDataset
from hdpy.model import train_mlp, val_mlp
from hdpy.ecfp import compute_fingerprint_from_smiles
SCRATCH_DIR = "/p/vast1/jones289"

# args contains things that are unique to a specific run
hdc_parser = hdc_args.get_parser()
hdc_parser.add_argument('--mode', choices=['encode', 'test'])
hdc_parser.add_argument('--output-prefix', default="debug")
args = hdc_parser.parse_args()
# config contains general information about the model/data processing
config = hdc_args.get_config(args)

def perf_trial():

    if args.mode == 'encode':

        # run encode loop
        dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # persistent_workers=True,
        shuffle=False,
        collate_fn=collate_fn,
        )

        if model.name == "mlp" or config.embedding == "directecfp":
            # compute ecfps and return mean time and std
            time_list = []
            for smiles in tqdm(np.concatenate([train_smiles, test_smiles])):
                _, ecfp_time = compute_fingerprint_from_smiles(smiles=smiles, length=config.ecfp_length, radius=config.ecfp_radius, return_time=True)
                time_list.append(ecfp_time)

            mean_encode_time = np.mean(time_list)
            std_encode_time = np.std(time_list)

        else: 

            encodings, labels, encode_time_arr = encode_hdc(model=model, dataloader=dataloader, device=device, use_numpy=True)

            mean_encode_time = encode_time_arr.mean()
            std_encode_time = encode_time_arr.std()
        if output_target_encode_path.exists():
            pass
        elif model.name == "mlp" or config.embedding == "directecfp":
            pass
        else:
            torch.save((encodings, labels), output_target_encode_path)

    elif args.mode == 'test':


        # run test loop
        dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # persistent_workers=True,
        shuffle=False,
        collate_fn=collate_fn,
        )

        if model.name == "mlp":
            test_result = val_mlp(model=model, val_dataloader=dataloader, device=device)
        else:
            test_result = test_hdc(model=model, test_dataloader=dataloader, device=device, encode=False) 
        
        torch.cuda.empty_cache()


    if args.mode == "encode":
        return mean_encode_time, std_encode_time

    elif args.mode in ["test", "mlp-test"]:

        return test_result['test_time_mean'], test_result['test_time_std']


def main():

    result_list = []

    for i in range(10):
        result_list.append(perf_trial())

    mean_time = np.mean([x[0] for x in result_list])
    std_time = np.std([x[1] for x in result_list])
    print(f"mean (avg): {mean_time}, std (avg): {std_time}")

    output_path = Path(f"{args.output_prefix}_{args.mode}.npy")
    np.save(output_path, np.array([len(dataset), mean_time, std_time]))

if __name__ == "__main__":

    dataset = None
    model = get_model(config)



    if config.device == 'cpu':
        device = 'cpu'
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # print(f'using device {device}')

    if args.mode == "test":
        model.to(device)
        if model.name != "mlp":
        # this should be addressed in a better way such as registering as a nn submodule
            model.am = model.am.to(device)

    root_data_p = Path("/p/vast1/jones289/molformer_embeddings/molnet/hiv/")
    if not root_data_p.exists():
        print(f"{root_data_p} does not exist yet. creating.")
        root_data_p.mkdir(parents=True)


    collate_fn, encodings, labels = None, None, None
    # train_dataset, test_dataset = None, None
    dataloader = None
    if config.model == "molehd":
        collate_fn = collate_list_fn



    train_molformer_path = root_data_p / Path(
                "train_N-Step-Checkpoint_3_30000.npy"
                )

    test_molformer_path = root_data_p / Path(
    "test_N-Step-Checkpoint_3_30000.npy"
    )
        
    train_smiles_path = root_data_p / Path(
                "train_N-Step-Checkpoint_3_30000_smiles.npy"
                )

    test_smiles_path = root_data_p / Path(
    "test_N-Step-Checkpoint_3_30000_smiles.npy"
    )

    output_target_encode_path = Path(f"/p/vast1/jones289/hdbind/molnet/hiv/{config.model}_{config.embedding}_{config.D}_cache.pt")

    if not output_target_encode_path.parent.exists():
        output_target_encode_path.parent.mkdir(parents=True, exist_ok=True)


    train_smiles = np.load(train_smiles_path, allow_pickle=True)
    test_smiles = np.load(test_smiles_path, allow_pickle=True)
    train_data = np.load(train_molformer_path)
    test_data = np.load(test_molformer_path)

    x_train = train_data[:, :-1] 
    y_train = train_data[:, -1]

    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]       


    if config.embedding == "molformer":

        train_dataset = TensorDataset(
            torch.from_numpy(x_train).float(),
            torch.from_numpy(y_train).int(),
        )
        test_dataset = TensorDataset(
            torch.from_numpy(x_test).float(),
            torch.from_numpy(y_test).int(),
        )
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    elif config.embedding == "directecfp":

        
        train_dataset = StreamingECFPDataset(smiles_list=train_smiles, 
                                                labels=y_train, 
                                                length=config.ecfp_length, 
                                                radius=config.ecfp_radius)
        test_dataset = StreamingECFPDataset(smiles_list=test_smiles, 
                                                labels=y_test, 
                                                length=config.ecfp_length, 
                                                radius=config.ecfp_radius)
        
        dataset= torch.utils.data.ConcatDataset([train_dataset, test_dataset])


    elif config.embedding == "molformer-decfp-combo":

        train_dataset = StreamingComboDataset(smiles_list=train_smiles,
                                                    feats=x_train,
                                                    labels=y_train, 
                                                    length=config.ecfp_length, 
                                                    radius=config.ecfp_radius)
        test_dataset = StreamingComboDataset(smiles_list=test_smiles,
                                                    feats=x_test, 
                                                    labels=y_test, 
                                                    length=config.ecfp_length, 
                                                    radius=config.ecfp_radius)


        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    elif config.embedding == "ecfp":
        train_dataset = ECFPFromSMILESDataset(
                    smiles=train_smiles,
                    labels=y_train,
                    ecfp_length=config.ecfp_length,
                    ecfp_radius=config.ecfp_radius,
                )

        test_dataset = ECFPFromSMILESDataset(
            smiles=test_smiles,
            labels=y_test,
            ecfp_length=config.ecfp_length,
            ecfp_radius=config.ecfp_radius,
        )

        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    if args.mode == "test":
        if model.name == "mlp":
            pass
        else:
            if config.embedding == "directecfp":
                print(f"using {config.embedding} does not require encoding. passing encode check.")
            elif output_target_encode_path.exists():
                encodings, labels = torch.load(output_target_encode_path)
                # load the encodings and labels
                dataset = torch.utils.data.TensorDataset(encodings, labels)
            else:
                raise RuntimeError("run encode first")

    main()