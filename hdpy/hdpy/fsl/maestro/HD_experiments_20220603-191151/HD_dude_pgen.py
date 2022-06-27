"""Produces a custom ParameterGenerator for DUD-E dataset."""
from itertools import product
from maestrowf.datastructures.core import ParameterGenerator


def get_custom_generator(env, **kwargs):
    """
    Create a custom populated ParameterGenerator with a single Parameter.
    :params env: A StudyEnvironment object containing custom information.
    :params kwargs: A dictionary of keyword arguments this function uses.
    :returns: A ParameterGenerator populated with parameters.
    """
    p_gen = ParameterGenerator()

    target_list = [
        "ace",
        "aces",
        "ada",
        "aldr",
        "ampc",
        "andr",
        "bace1",
        "cdk2",
        "comt",
        "dyr",
        "egfr",
        "esr1",
        "fa10",
        "gcr",
        "hivpr",
        "hivrt",
        "hmdh",
        "hs90a",
        "inha",
        "kith",
        "mcr",
        "mk14",
        "nram",
        "parp1",
        "pde5a",
        "pgh1",
        "pgh2",
        "pnph",
        "pparg",
        "prgr",
        "pur2",
        "pygm",
        "rxra",
        "sahh",
        "src",
        "thrb",
        "try1",
        "vgfr2",
    ]

    feat_list = [
        "ecfp",
        "smiles_to_seq",
        "smiles_to_image",
        "mordred",
        "maacs",
        "rdkit",
    ]

    param_space_list = list(product(target_list, feat_list))

    param_dict = {
        "TARGET": {
            "values": [x[0] for x in param_space_list],
            "label": "TARGET.%%",
        },
        "FEAT": {
            "values": [x[1] for x in param_space_list],
            "label": "FEAT.%%",
        },
    }
    for key, value in param_dict.items():
        p_gen.add_parameter(key, value["values"], value["label"])

    return p_gen
