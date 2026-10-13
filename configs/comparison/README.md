# Comparison experiment configs

No comparison model is integrated yet, so this directory contains no runnable model configs. Add one config when the corresponding official source and adapter are present.

For each dataset/model/seed, record the external source revision, constructor and checkpoint source, explicit input and output contracts, the matched split and 224x224 evaluation protocol, the random seed, and the actual optimizer/loss/schedule used. Do not fill missing official hyperparameters with guesses or treat paper-reported metrics as matched runs.

The existing proposed configs remain at `configs/octa500_3m.yaml`, `configs/octa500_6m.yaml`, and `configs/rossa.yaml`.
