# Comparison experiment configs

The six copied OCTAMamba model-zoo sources have constructor and output adapters,
and a shared OCTA500-3M training runner at comparison_models/train_3m.py.
The runner reads the frozen configs/octa500_3m.yaml protocol; this directory
still contains no separate model-specific official training configs.

For each dataset/model/seed, record the external source revision, constructor and
checkpoint source, explicit input and output contracts, the matched split and
224x224 evaluation protocol, the random seed, and the actual optimizer/loss/schedule
used. Do not fill missing official hyperparameters with guesses or treat
paper-reported metrics as matched runs.

An adapter config must include `image_size: [224, 224]` and
`model: {name: <registered model name>}`. The proposed-model configs remain at
`configs/octa500_3m.yaml`, `configs/octa500_6m.yaml`, and `configs/rossa.yaml`.
