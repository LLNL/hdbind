# this is the root directory for hdpy

<!-- Use `main.py` as the main driver program -->

<!-- `fsl` contains general code for HDC -->

<!-- `rff-hdc` contains code for VSA variants -->

<!-- in order to install, please use  -->
<!-- > -->
<!-- ... -->


<!-- to run the example: -->
<!-- ``` -->
<!-- python main.py --dataset dude --out-csv dude-rff-gvsa --model-list rff-gvsa --train-path-list ../../datasets/dude/deepchem_feats/try1/ecfp/train.npy --test-path-list ../../datasets/dude/deepchem_feats/try1/ecfp/test.npy --n-problems 1 --hidden-size 2048 --out-data-dir dude-rff-gvsa -->
<!--  -->
<!-- ``` -->

The main file to run is ```hd_main.py```. The file ```hd_model.py``` contains details for the abstract HD implementation that is further specified by the ```mole_hd/``` and ```ecfp_hd/``` modules.

the following bash scripts run the benchmarks:

	* run_moleculenet.sh
	* run_litpcba.sh
	* run_dude.sh