import torch 
from thop import profile
from hdpy.model import RPEncoder, MLPClassifier
from deepchem.molnet import load_hiv, load_tox21, load_bace_classification, load_sider
from argparse import Namespace
from hdpy.data_utils import ECFPFromSMILESDataset, SMILESDataset
from pathlib import Path
import deepchem as dc
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.preprocessing import normalize
from torchstat import stat


SCRATCH_DIR = "/p/vast1/jones289"

def main(config, args, dim_frac):
    

    smiles_featurizer = dc.feat.DummyFeaturizer()

    # input("enter any key to continue")

    if args.dataset == "hiv":
        dataset = load_hiv(splitter="scaffold", featurizer=smiles_featurizer)
        target_list = dataset[0]

        # use something besides train_dataset/test_dataset?
        train_dataset = dataset[1][0]
        test_dataset = dataset[1][1]

        smiles_train = train_dataset.X
        y_train = train_dataset.y

        smiles_test = test_dataset.X
        y_test = test_dataset.y


    
    n_comps = int(768 * dim_frac) 
    
    if config.model == "rp":
        model = RPEncoder(input_size=n_comps, 
                              D=config.D, 
                              num_classes=2)

        model = model.half()

    elif config.model == "mlp":
        model = MLPClassifier(layer_sizes=((n_comps, 512), (512, 256), (256, 128), (128, 2)),
                                lr=1e-3, 
                                activation=torch.nn.GELU(),
                                criterion=torch.nn.NLLLoss(),
                                optimizer=torch.optim.Adam)
 

    torch.cuda.reset_peak_memory_stats(device=None)
    print(f"gpu used {torch.cuda.max_memory_allocated(device=None)} memory")

    model.to(config.device)

    torch.cuda.reset_peak_memory_stats(device=None)
    print(f"gpu used {torch.cuda.max_memory_allocated(device=None)} memory")



    roc_values = []
    std_values = []
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

        elif config.embedding == "molformer":

            train_data = np.load(f"{SCRATCH_DIR}/molformer_embeddings/molnet/{args.dataset}/train_N-Step-Checkpoint_3_30000.npy")
            test_data = np.load(f"{SCRATCH_DIR}/molformer_embeddings/molnet/{args.dataset}/test_N-Step-Checkpoint_3_30000.npy")

            x_train = train_data[:, :768]
            x_test = test_data[:, :768]

            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            pca = PCA()

            scaler = StandardScaler()
            scaler.fit(x_train)

            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)

            x_train = x_train[:, :n_comps]
            x_test = x_test[:, :n_comps]

            y_train = train_data[:, (768+target_idx)]

            y_test = test_data[:, (768+target_idx)]

            train_dataset = TensorDataset(torch.from_numpy(x_train).half(), 
                                          torch.from_numpy(y_train).half())
            test_dataset = TensorDataset(torch.from_numpy(x_test).half(), 
                                         torch.from_numpy(y_test).half())
            

        elif config.embedding == "molclr":

            # we're just using the GIN model always 

            train_data = np.load(f"{SCRATCH_DIR}/molclr_embeddings/molnet/{args.dataset}/train_{target_name}.npy")
            test_data = np.load(f"{SCRATCH_DIR}/molclr_embeddings/molnet/{args.dataset}/test_{target_name}.npy")


            train_dataset = TensorDataset(torch.from_numpy(normalize(train_data[:, :-1], norm="l2", axis=0)).float(), 
                                          torch.from_numpy(train_data[:, -1]).float())
            test_dataset = TensorDataset(torch.from_numpy(normalize(test_data[:, :-1], norm="l2", axis=0)).float(), 
                                         torch.from_numpy(test_data[:, -1]).float())




        loader = DataLoader(dataset=torch.utils.data.ConcatDataset([train_dataset, test_dataset]),
                            batch_size=1,
                            num_workers=args.num_workers,
                            shuffle=False,
                            drop_last=False,
                        )
        from tqdm import tqdm 

        macs_list = []
        params_list = []
        for batch in tqdm(loader):
            # import pdb
            # pdb.set_trace()
            if config.model == "mlp":
                batch = batch[0].reshape(1,1,-1).float()
            else:
                batch = batch[0].reshape(1,-1)
            macs, params = profile(model, inputs=batch.to(config.device))
            # stat(model, batch)
            print(macs, params)
            macs_list.append(macs)
            params_list.append(params)

        # print(np.mean(macs_list), np.mean(params_list))
        print(f"{exp_name} macs (mean) (std): {np.mean(macs_list)} ({np.std(macs_list)})")
        print(f"{exp_name} params (mean) (std): {np.mean(params_list)} ({np.std(params_list)})")



        from hdpy.main_molnet import main as molnet_main

        result_dict = molnet_main(args=args, config=config, model=model, train_dataset=train_dataset, test_dataset=test_dataset)
        roc_values.append(np.mean([value["roc-auc"] for value in result_dict["trials"].values()]))
        std_values.append(np.std([value["roc-auc"] for value in result_dict["trials"].values()]))

    print(f"Average ROC-AUC is {np.mean(roc_values)} +/- ({np.mean(std_values)})")


if __name__ == "__main__":
    import argparser

    # args contains things that are unique to a specific run
    args = argparser.parse_args()

    if args.split_type != "scaffold":
        print(f"{args.split_type} not supported for this dataset! please use scaffold for molnet")
        assert args.split_type == "scaffold"
    # config contains general information about the model/data processing
    config = argparser.get_config(args)

    if config.device == "cpu":
        device = "cpu"
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"using device {device}")

    exp_name = f"{Path(args.config).stem}"

    for dim_frac in [0.01, .1, .5, .75, 1.0]:
        main(config=config, args=args, dim_frac=dim_frac)
