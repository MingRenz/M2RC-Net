# External comparison model framework

This package is separate from `models/`, which contains the proposed CAMS-MBRCNet family. No external model is integrated or registered in this commit. Planned integrations include OCTAMamba, HV-OCTAMamba, DGNet, and PMDC-Net; each will arrive in its own later commit with source attribution and an adapter.

## Add one comparison model

1. Add `comparison_models/<ModelName>/` containing the official model code, its license/source revision, and a small `adapter.py`.
2. Implement a builder accepting a config and returning `ComparisonModelBundle(model, adapter)`.
3. Register its lazy import in `comparison_models/build.py` with `register_comparison_model("model_name", "comparison_models.<ModelName>.adapter:build")`.
4. Add a model-specific config under `configs/comparison/` after verifying the official constructor, training objective, checkpoints, and input/output conventions.
5. Test import, weight loading, forward, backward, and the common data/evaluation protocol before any formal run.

`ComparisonAdapter.extract_output` returns `SegmentationOutput` with the original prediction tensor, an explicit `logits` or `probability` label, and auxiliary outputs. The framework never silently applies sigmoid, resizing, thresholding, or a replacement loss. A future training runner must handle model-specific official losses explicitly.

The current proposed-model `train.py`, `test.py`, `models.build_model`, and checkpoint logic remain untouched. No comparison runner is enabled until the first real comparison model is integrated.

## Run paths

Future runs use `log/<dataset>/<proposed|comparison>/<model>/seed_<N>/`. `reserve_seed_log_dir` refuses to overwrite an existing seed directory. The historical `log/<dataset>/<variant>/run_NNN/` records are kept in place. Aggregated statistics belong in `results/<dataset>/`, never in a training run directory. Both `log/` and `results/` are Git ignored.
