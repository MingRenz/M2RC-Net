# External comparison model framework

This package is separate from `models/`, which contains the proposed CAMS-MBRCNet family. Six OCTAMamba model-zoo source copies and the original OCTAMamba and HV-OCTAMamba models have thin, lazily registered adapters. A shared OCTA500-3M comparison runner now exists at comparison_models/train_3m.py, with launcher scripts/train_comparison_3m.sh. It uses the frozen Base data, Dice loss, optimizer, validation metric, 400-epoch cap, and patience 40. These are common-protocol random-initialization comparisons, not reproductions of each model's official training recipe or checkpoint. Planned later integrations include DGNet and PMDC-Net.

## Copied source adapters

| Registry name | Copied source | Constructor | Raw binary output |
| --- | --- | --- | --- |
| `unet` | `unet/Unet.py` | `Unet(n_class=1)` | sigmoid probability |
| `unetpp` | `unetpp/unetpp.py` | `ResNet34UnetPlus(num_channels=1, num_class=1)` | sigmoid probability |
| `swin_unet` | `swin_unet/Swin_UNet.py` | `SwinUnet(img_size=224, num_classes=1)` | sigmoid probability |
| `h2former` | `h2former/H2Former.py` | `res34_swin_MS(image_size=224, num_class=1)` | sigmoid probability |
| `vm_unet` | `vm_unet/VM_UNet.py` | `VMUNet(input_channels=1, num_classes=1)` | sigmoid probability |
| `h_vmunet` | `h_vmunet/H_vmunet.py` | `H_vmunet(num_classes=1, input_channels=1)` | sigmoid probability |
| `octamamba` | `octamamba/OCTAMamba.py` | `OCTAMamba()` | sigmoid probability |
| `hv_octamamba` | `hv_octamamba/HV_OCTAMamba.py` | `HV_OCTAMamba()` | sigmoid probability |

The original six model-zoo source files are byte-for-byte copies from the local OCTAMamba `modelszoo` directory; their original upstream revisions and official training protocols have not been independently verified. Each adapter accepts an explicit comparison config with `image_size: [224, 224]` and `model: {name: <registry name>}`. Other model options are rejected rather than silently ignored. Input remains one-channel 224x224; output remains the source model's probability tensor. The adapter does not apply sigmoid, interpolation, thresholding, or a loss.

No pretrained weights are loaded by these constructors. In particular, the copied VM-UNet and H-VMUNet require a working selective-scan implementation for forward execution. Source attribution, checkpoint choices, objective, optimizer, and dependency/forward checks must be settled before a matched comparison run.

## Add one comparison model

1. Add `comparison_models/<ModelName>/` containing the official model code, its license/source revision, and a small `adapter.py`.
2. Implement a builder accepting a config and returning `ComparisonModelBundle(model, adapter)`.
3. Register its lazy import in `comparison_models/build.py` with `register_comparison_model("model_name", "comparison_models.<ModelName>.adapter:build")`.
4. Add a model-specific config under `configs/comparison/` after verifying the official constructor, training objective, checkpoints, and input/output conventions.
5. Test import, weight loading, forward, backward, and the common data/evaluation protocol before any formal run.

`ComparisonAdapter.extract_output` returns `SegmentationOutput` with the original prediction tensor, an explicit `logits` or `probability` label, and auxiliary outputs. The framework never silently applies sigmoid, resizing, thresholding, or a replacement loss. The current 3M runner deliberately uses the Base Dice loss on the copied models' probability outputs. A future official-protocol reproduction would need separately verified model-specific losses and checkpoints.

The current proposed-model `train.py`, `test.py`, `models.build_model`, and checkpoint logic remain untouched. The separate comparison runner does not modify these proposed-model entry points. It fixes the seed and DataLoader order but disables torch.use_deterministic_algorithms(True), since four copied models fail CUDA bilinear backward under the current PyTorch build and VM-UNet has shown gradient differences across same-seed processes.

## Run paths

Comparison runs use `log/<dataset>/comparison/<model>/seed_<N>/run_001`, `run_002`, and so on. The launcher allocates the next run directory atomically and never overwrites a previous run. The historical `log/<dataset>/<variant>/run_NNN/` records are kept in place. Aggregated statistics belong in `results/<dataset>/`, never in a training run directory. Both `log/` and `results/` are Git ignored.

## Multi-dataset comparison runs

The 6M and ROSSA launchers are `scripts/train_comparison_6m.sh MODEL [SEED]` and `scripts/train_comparison_rossa.sh MODEL [SEED]`. They reuse the registered model adapters and the 3M comparison loss, optimizer, checkpoint selection, and evaluation protocol through `comparison_models/train_multidataset.py`. The shared data readers provide 180/20/100 samples for 6M and 718/100/100 for ROSSA, including both ROSSA training subsets. Each launch creates `log/<dataset>/comparison/<model>/seed_<N>/run_NNN` without overwriting previous runs. Only one comparison training process is allowed at a time across the three datasets.
