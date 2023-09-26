import argparse
import yaml
from yaml.loader import SafeLoader


def get_parser():
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
    # use the model argument to lookup the respective config file
    # parser.add_argument("--model", choices=["smiles-pe", "selfies", "ecfp", "rp", "rf", "mlp"])
    parser.add_argument("--config", help="path to config file containing model information")
    parser.add_argument(
        "--n-trials", type=int, default=1, help="number of trials to perform"
    )
    # parser.add_argument("--dry-run", action="store_true")
    # parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--random-state", type=int, default=0)


    return parser

def parse_args():

    parser = get_parser()


    args = parser.parse_args()
    print(f"args: {args}")
    return args


def get_config(args):

    with open(args.config, 'r') as f:

        config = yaml.load(f, Loader=SafeLoader)
        config = argparse.Namespace(**config)
        print(f"config: {config}")
    return config

