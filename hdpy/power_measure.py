import torch
import numpy as np
from hdpy.model import get_model, train_hdc, test_hdc, encode_hdc 
from torch.utils.data import TensorDataset
import hdpy.hdc_args as hdc_args
from tqdm import tqdm 
from hdpy.utils import collate_list_fn
from torch.utils.data import DataLoader
from pathlib import Path
import subprocess
SCRATCH_DIR = "/p/vast1/jones289"



# '''
# args contains things that are unique to a specific run
hdc_parser = hdc_args.get_parser()
hdc_parser.add_argument('--mode', choices=['encode', 'train', 'test', 'mlp-train', 'mlp-test'])
# hdc_parser.add_argument('--perf-output', default="perf_profile.csv")
args = hdc_parser.parse_args()
# config contains general information about the model/data processing
config = hdc_args.get_config(args)

# collect GPU statistics
# causes issues with shell=True which is apparently required for passing arguments? and gives a too many open files error when running
# subprocess.run([f"nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory,temperature.gpu,pstate, --format=csv -l 1 -f {args.perf_output}&"], shell=True)



model = get_model(config)



if config.device == 'cpu':
    device = 'cpu'
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f'using device {device}')
model.to(device)
if config.model in ["rp"]:
# this should be addressed in a better way such as registering as a nn submodule
    model.am = model.am.to(device)

# model = torch.nn.DataParallel(model)

print(model)

encode_time_list = []
train_time_list = []
test_time_list = []

lit_pcba_ave_p = Path("/p/vast1/jones289/lit_pcba/AVE_unbiased")

# target_list = list(lit_pcba_ave_p.glob("*/"))
target_list = list(lit_pcba_ave_p.glob("VDR*/"))
for target_path in tqdm(target_list):

    target_name = target_path.name

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


    if config.embedding == "ecfp":
        train_molformer_path = lit_pcba_ave_p / Path(target_name) / Path("ecfp_train.npy") 
        
        test_molformer_path = lit_pcba_ave_p / Path(target_name) / Path("ecfp_test.npy") 

    collate_fn, encodings, labels = None, None, None
    if config.model == "molehd":
        collate_fn = collate_list_fn

    
    output_target_encode_path = Path(f"/p/vast1/jones289/hdbind/lit-pcba/{target_name}_{config.model}_{config.embedding}_{config.D}_cache.pt")

    if args.mode == 'encode':
        train_data = np.load(train_molformer_path)
        test_data = np.load(test_molformer_path)

        x_train = train_data[:, :-1] 
        y_train = train_data[:, -1]

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
        # run encode loop
        dataloader = DataLoader(
        torch.utils.data.ConcatDataset([train_dataset, test_dataset]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=True,
        collate_fn=collate_fn,
        )
        encodings, labels, encode_time = encode_hdc(model=model, dataloader=dataloader, device=device)

        # maybe do the actual generating and loading separately so there's no risk of unneccessary gpu-cpu data transfer

        if output_target_encode_path.exists():
            pass
        else:
            torch.save((encodings, labels), output_target_encode_path)

        print(f"encode: N={len(labels)}, time={encode_time} ({encode_time/len(labels)} s\mol)")

        encode_time_list.append((encode_time, len(labels)))

    elif args.mode == 'train':

        if not output_target_encode_path.exists():
            raise RuntimeError("run encode first")
        # load the encodings and labels
        encodings, labels = torch.load(output_target_encode_path)
        dataset = torch.utils.data.TensorDataset(encodings, labels)

        # run train loop
        dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=True,
        collate_fn=collate_fn,
        )
        train_result = train_hdc(model=model, train_dataloader=dataloader, device=device, num_epochs=1,
                  encode=False) 


        print(f"train: N={len(labels)}, time-AM={train_result[2]} ({train_result[2]/ len(labels)} s/mol), time-retrain={train_result[3]} ({train_result[3] / len(labels)} s/mol)")

        
        train_time_list.append((train_result[2], train_result[3], len(labels)))

    elif args.mode == 'test':
        if not output_target_encode_path.exists():
            raise RuntimeError("run encode first")
        encodings, labels = torch.load(output_target_encode_path)
        # load the encodigns and labels

        dataset = torch.utils.data.TensorDataset(encodings, labels)
        # run test loop
        dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=True,
        collate_fn=collate_fn,
        )
        test_result = test_hdc(model=model, test_dataloader=dataloader, device=device, encode=False) 

        print(f"test: N={len(labels)}, time={test_result['test_time']} ({test_result['test_time']/len(labels)} s/mol)")


        test_time_list.append((test_result['test_time'], len(labels)))


    elif args.mode == "mlp-train":
        from hdpy.model import train_mlp
        
        train_data = np.load(train_molformer_path)

        x_train = train_data[:, :-1] 
        y_train = train_data[:, -1]

        train_dataset = TensorDataset(
            torch.from_numpy(x_train).float(),
            torch.from_numpy(y_train).int(),
        )

        dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=True,
        collate_fn=collate_fn,
        )
        train_result = train_mlp(model=model, train_dataloader=dataloader, 
                  epochs=1, device=device)
        
        print(f"train: N={y_train.shape[0]}, time={train_result['train_time']} ({train_result['train_time']/y_train.shape[0]} s/mol)")


        train_time_list.append((train_result['train_time'], y_train.shape[0] ))

    elif args.mode == "mlp-test":
        from hdpy.model import val_mlp
        test_data = np.load(test_molformer_path)

        x_test = test_data[:, :-1]
        y_test = test_data[:, -1]

        test_dataset = TensorDataset(
            torch.from_numpy(x_test).float(),
            torch.from_numpy(y_test).int(),
        )
        # run encode loop
        dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=True,
        collate_fn=collate_fn,
        )
        test_result = val_mlp(model=model, val_dataloader=dataloader, device=device)

        print(f"test: N={y_test.shape[0]}, time={test_result['forward_time']} ({test_result['forward_time']/y_test.shape[0]} s/mol)")

        test_time_list.append((test_result['forward_time'], y_test.shape[0]))

    del encodings
    del labels
    torch.cuda.empty_cache()


if args.mode == "encode":
    print(f"time-encode (mean s/mol): {np.mean([x[0] / x[1] for x in encode_time_list])}")

elif args.mode == "train":
    print(f"time-am (mean s/mol): {np.mean([x[0] / x[2] for x in train_time_list])}")
    print(f"time-retrain (mean s/mol): {np.mean([x[1] / x[2] for x in train_time_list])}")
elif args.mode == "mlp-train":
    print(f"time-train (mean s/mol): {np.mean([x[0] / x[1] for x in train_time_list])}")
elif args.mode in ["test", "mlp-test"]:
    print(f"time-test (mean s/mol): {np.mean([x[0] / x[1] for x in test_time_list])}")