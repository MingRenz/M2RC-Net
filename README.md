# CAMS-MBRCNet research code

Stable Clean SSM Backbone + CAMS (Caliber-Aware Multi-Scale State-Space Module) + MBRC (Multistage Bidirectional Residual Correction) = CAMS-MBRCNet.

## Variants

base = Backbone; cams = Backbone+CAMS; mbrc = Backbone+MBRC; cams_mbrcnet = CAMS-MBRCNet. The unified builder instantiates only enabled modules. Existing model attribute names and checkpoint keys are retained.

## Datasets

configs/octa500_3m.yaml, configs/octa500_6m.yaml, and configs/rossa.yaml specify the frozen 224×224 split, preprocessing, optimization, and evaluation settings. data/ contains copied datasets and precomputed CAMS maps on this server. CAMS maps are read only.

## Entry points

Run bash scripts/train_3m.sh base (or cams, mbrc, cams_mbrcnet) and similarly train_6m.sh / train_rossa.sh. Each run writes a new log/<dataset>/<variant>/run_NNN directory. Run python test.py --config ... --variant ... --checkpoint ... to evaluate one saved checkpoint.

The checkpoint criterion is mean per-image Val Dice after probability min-max normalization, uint8 conversion and Otsu thresholding. Base and CAMS test their best checkpoint at training end; MBRC variants retain their established test-on-new-best behavior. Tests perform only in-memory one-batch optimizer steps; no formal training is launched.

tests/equivalence.py compares the four variants against /home/zfzf/final_final under the same A40 software environment. utils/checkpoint.py loads the three existing outer checkpoint formats with strict model-key checking.

The launch scripts return PID, RUN and LOG. Inspect a running experiment with tail -n 80 -F /path/to/console.log; tqdm updates are streamed through the same file.

## Source and verification

The original /home/zfzf/final_final is retained unchanged. tests/source_manifest.json records its code/launcher SHA256 manifest. The backbone is copied byte for byte; CAMS and MBRC definitions are merged once with import-path edits only. tests/equivalence.py compares initial state, output, CAMS stage features, losses, every gradient, Adam state and updated parameters for all four variants. tests/data_smoke.py checks source-vs-new ordered splits and real batches. tests/real_full_smoke.py checks one full-model step on all three datasets. The reports in tests/ are diagnostics from untrained initial models, not formal performance results.

## 2080 Ti copy

This copy is located at /home/zzf/myeye/CAMS-MBRCNet and uses /home/zzf/miniconda3/envs/octa_repro/bin/python. The historical source-equivalence tests reference A40 paths and remain as archived validation records. No training has been started by the transfer.

## Comparison framework

`comparison_models/` contains the independent lazy registry, adapter contract, and future seed-run path rules for external paper baselines. No external model is registered yet. See `comparison_models/README.md` and `configs/comparison/README.md`. The existing proposed-model training entry points and historical logs remain unchanged.
