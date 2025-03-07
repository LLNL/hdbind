# hdpy 

Repository for *HD-Bind: Encoding of Molecular Structure with Low Precision, Hyperdimensional Binary Represenations* by Derek Jones, Xiaohua Zhang, Brian J. Bennion, Sumukh Pinge, Weihong Xu, Jaeyoung Kang, Behnam Khaleghi, Niema Moshiri, Jonathan E. Allen and Tajana S. Rosing


Link to manuscript [here](https://www.nature.com/articles/s41598-024-80009-w)

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



To install hdpy (from root directory):

> conda create --name hdpy --file hdpy_env_release.yml

> python -m pip install . 





## Separately installing dependencies

Separately you can do the following: 

Install the [deepchem](https://github.com/deepchem/deepchem) library 

#> conda install -c conda-forge deepchem #using conda but can refer to the docs for your specific install 
> pip install --pre deepchem #conda install doesn't work currently, use nightly build

Next, install [PyTorch](https://pytorch.org/). This project does not make use of torchvision or torchaudio so we'll skip that (feel free to do so if inclined)
> conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

Next, install [rdkit](https://www.rdkit.org/docs/Install.html#cross-platform-using-conda)
> conda install -c conda-forge rdkit


Next, ray 
> pip install ray==2.7.0rc0 

Next, SmilesPE
> pip install SmilesPE 

Next, SELFIES

> pip install selfies






## Running the benchmarks

To run the [MoleculeNet](https://moleculenet.org/) training and testing script:

> python main_molnet.py --dataset bbbp --split-type scaffold --n-trials 10 --random-state 5 --batch-size 128 --num-workers 8 --config configs/hdbind-rp-ecfp-1024-1.yml


To run the [LIT-PCBA](https://drugdesign.unistra.fr/LIT-PCBA/) training and testing script:

> python main_litpcba.py --dataset lit-pcba --split-type ave --n-trials 10 --random-state 5 --batch-size 128 --num-workers 8 --config configs/hdbind-rp-ecfp-1024-1.yml


# Getting Involved

Contact Derek Jones for any questions/collaboration to expand the project! djones@llnl.gov, wdjones@ucsd.edu


## Citation

Jones, D., Zhang, X., Bennion, B. J., Pinge, S., Xu, W., Kang, J., Khaleghi, B., Moshiri, N., Allen, J. E., & Rosing, T. S. (2024). HDBind: encoding of molecular structure with hyperdimensional binary representations. Scientific Reports, 14(1), 1-16.. https://doi.org/10.1038/s41598-024-80009-w
