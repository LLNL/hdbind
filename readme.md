# HDPY

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


## Citation

Jones, D., Allen, J. E., Zhang, X., Khaleghi, B., Kang, J., Xu, W., Moshiri, N., & Rosing, T. S. (2023, March 27). HD-Bind: Encoding of Molecular Structure with Low Precision, Hyperdimensional Binary Representations. arXiv. http://arxiv.org/abs/2303.15604
