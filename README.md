# SLCP
This repository is an implementation of the submission **Split Localized Conformal Prediction**. 

To set up the environment, run:
```
conda env create -f conformal.yml
```
then activate the environment:
```
conda activate conformal
```

To run all experiments mentioned in the paper, first change UtilsParams.experiment in [config.py](./SLCP/config.py) into corresponding name of the experiment (available choice: "prediction", "cov_shift", "toy_plot", "nn_capacity"), then run
```
python -W ignore run_all_experiments.py
```
The results will be saved into a CSV file.