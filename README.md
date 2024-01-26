# hdpy 

Repository for ``HD-Bind: Encoding of Molecular Structure with Low Precision, Hyperdimensional Binary Representations'' by Derek Jones, Jonathan E. Allen, Xiaohua Zhang, Behnam Khaleghi, Jaeyoung Kang, Weihong Xu, Niema Moshiri, Tajana S. Rosing

- ecfp/: contains implementations of ecfp encoding algorithms
- molehd/: contains implementations of the MoleHD (Ma et.al) SMILES-based encoding algorithms
- prot_lig/: contains implementations of HDC encoding for protein drug interactions
- selfies/: contains implementaions of encoding algorithms for SELFIES strings
- configs/: contains configuration files for the various HDC models

- argparser.py: contains logic for the arguments used to drive the programs in this project
- data_utils.py: contains logic for dataloading 
- encode_utils.py: contains general encoding logic
- main.py: driver program for HDBind experiments
- metrics.py: contains logic for the various metrics used in the work
- model.py: contains logic for the HDC model implementations themselves
- run_timings.py: contains logic to estimate timing information for various processes such as ECFP computation
- sdf_to_smiles.py: utility script to convert collections of molecules
- utils.py: additional utility functions



# Getting started

In order to install the required dependencies, please first install [anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) or [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).


To install the hdpy repository

> conda create --name hdpy --file hdpy_env.yml

To run the [MoleculeNet] training and testing script:

> python main_molnet.py --dataset tox21 --split-type scaffold --n-trials 10 --random-state 5 --batch-size 128 --num-workers 8 --config configs/hdbind-rp-molformer.yml

To run the [LIT-PCBA] training and testing script:

> python main_lit_pcba.py --dataset lit-pcba --split-type ave --n-trials 10 --random-state 5 --batch-size 128 --num-workers 8 --config configs/hdbind-rp-molformer.yml




# Getting Involved

Contact Derek Jones for any questions/collaboration to expand the project! djones@llnl.gov, wdjones@ucsd.edu


## Citation

Jones, D., Allen, J. E., Zhang, X., Khaleghi, B., Kang, J., Xu, W., Moshiri, N., & Rosing, T. S. (2023, March 27). HD-Bind: Encoding of Molecular Structure with Low Precision, Hyperdimensional Binary Representations. arXiv. http://arxiv.org/abs/2303.15604
