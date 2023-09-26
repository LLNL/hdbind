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

	target_list = ["mpro"]

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