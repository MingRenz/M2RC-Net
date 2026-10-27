# S3-OCTA single-projection adaptation with original core loss

- Model: S3-OCTA (S3OCTA).
- Official source path: /home/zzf/myeye/comparison_upstream/S3-OCTA (read-only reference).
- Original repository: supplied S3-OCTA-master.zip; its supplied README contains the paper link but no repository URL or Git revision metadata.
- Paper link supplied by upstream: https://doi.org/10.1016/j.bspc.2025.109117
- Model entry: funcs/model/S3OCTA.py:S3OCTA.
- Constructor: in_channels=3,out_channels=1,kernel_size=11,extend_scope=3,layer_depth=3,rate=72,ga="MiT",device_id="cpu". This retains the supplied train builder's effective settings: its config says extend_scope=7 but the actual builder omits it and therefore uses the constructor default 3.
- Project input: float32 [B,1,224,224], unchanged shared OCTA preprocessing.
- Model input adaptation: repeat the same grayscale channel three times to [B,3,224,224]. This is the authorized single-projection adaptation, NOT a recreation of three independent projections.
- Official input protocol difference: funcs/dataset/octa_500.py reads OCTA(FULL), OCTA(ILM_OPL), OCTA(OPL_BM) as three distinct input channels. Current project supplies one grayscale image. No extra projection data, CAMS maps or generated modality is introduced.
- Original output: one sigmoid probability [B,1,224,224]; no auxiliary output.
- Adapter output: the original probability unchanged; no repeated sigmoid, threshold, resizing or GT input.
- Parameters: 18,542,245 total/trainable (including the declared upstream blocks that are unused by its forward).
- Required dependencies: model uses torch, torchvision and einops; preserved loss helpers import OpenCV, NumPy and SciPy already installed in octa_repro. No new package was installed.
- Adaptation changes: package-relative MiT import; postponed annotations for Python 3.8; allow a CPU/qualified device string in the constructor; DSConv coordinate allocation follows offset.device so .to(device) remains valid.
- Architecture modifications: None. Keep all snake convolutions, bilinear interpolation, global MiT blocks, default channel widths and unused-but-declared official parameters.
- Auxiliary supervision: no model auxiliary heads are discarded. The original IBP_clDice objective is integrated separately in losses/s3_octa and selected lazily by losses/comparison.py.
- Training-specific notes: funcs/options/param_segment.py selects IBP_clDiceLoss; funcs/loss_function/clDice.py:IBP_clDice uses Dice and an IBP centerline term with equal 0.5 weights. funcs/custom/skeleton_extraction.py implements the Euler/boundary-peeling skeletonization. This is an explicit method component per upstream README, not merely an optimizer recipe.
- Current trainer compatibility: enabled through the shared comparison loss factory. S3-OCTA preserves original 0.5 Dice + 0.5 IBP-clDice; other comparison models keep the existing Dice by default. Model structure, optimizer, 224 input, splits, Otsu selection, logging and final-only Test policy remain unchanged.
- Approved protocol: one grayscale image repeated into three channels, plus the original core IBP objective. Report this as a single-projection adaptation, not an exact three-projection reproduction. Loss adaptation is explicitly documented in losses/s3_octa/README.md.
- Smoke verification: synthetic CPU [2,1,224,224] train-mode forward and original-core-loss backward passed with finite gradients; centerline term and offset convolutions both receive nonzero gradients. Batch-one loss/value/gradient/RNG match original; batch-two matches explicit independent original losses averaged. Shared evaluator runs and restores the training RNG. No optimizer update, formal training, real Test image or GPU-memory check was performed.

## Batch and random-stream compatibility

The supplied skeletonizer mixes batch elements for B>1 by treating the squeezed batch axis as Conv2d channels. Its official default batch size is 1. The new loss wrapper calls the unmodified batch-one objective per image and averages, supporting common batch size 2 without sample coupling. The actual upstream criterion(prediction,target) argument order, both random draws, STE, iteration count, epsilon and 0.5/0.5 weights are preserved. Evaluation uses a fixed diagnostic seed and restores Python, NumPy, CPU Torch and the selected CUDA RNG on exit. Training keeps the normal run seed stream.
