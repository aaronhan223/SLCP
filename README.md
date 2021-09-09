# LDCP
This repository is an implementation of the submission Conformal Prediction with Localized De-correlation. 

To set up the environment, run:
```
conda env create -f conformal.yml
```
then activate the environment:
```
conda activate conformal
```

To reproduce all results reported in the paper, under the [LDCP folder](./LDCP), run
```
python -W ignore run_all_experiments.py
```
The results will be saved into a CSV file.