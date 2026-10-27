# S3-OCTA core loss

Source: /home/zzf/myeye/comparison_upstream/S3-OCTA/funcs.
clDice.py changes only its two package imports. general.py and
skeleton_extraction.py are copied unchanged. Only IBP_clDice/Skeletonize_IBP
are used; unrelated upstream helpers are not invoked.

The source defaults are retained: alpha=0.5, clDice smooth=1.0, ordinary Dice
smooth=1e-6, IBP iterations=5, stochastic discretization and every STE.
The overwritten rand_like draw is intentionally retained to preserve RNG use.
The provided training code calls criterion(prediction,target); that exact call
order is retained despite the upstream forward parameter names.

Batch adaptation: the original skeletonizer squeezes [B,1,H,W] to [B,H,W]
and constructs Conv2d kernels with B input channels. For B>1 it combines samples.
The original training default is B=1. S3OCTALoss therefore evaluates each original
B=1 objective independently and returns their arithmetic mean. This is a
documented reduction adaptation, not equivalence to the broken batched code.

Inputs are probabilities and binary dataset masks. No extra sigmoid, threshold,
manual skeleton annotation or inference-time GT is introduced. Training retains
the normal seeded PyTorch stochastic stream. The shared evaluation context seeds
a separate diagnostic stream, restoring Python/NumPy/Torch CPU/current CUDA RNG
states on exit. Validation loss is diagnostic; formal Val Otsu Dice still selects
checkpoints. No source RNG calls or loss weights were removed or tuned.

Verification: synthetic CPU tests passed batch-one and batch-two reference
loss/gradient/final-RNG equality, independence of the first sample from changes
to the second, fixed diagnostic-seed replay, normal/exception RNG restoration,
and full S3 train-mode [2,1,224,224] forward/backward. Centerline and snake-offset
gradients were nonzero and finite. Shared evaluation passed. No formal training,
optimizer step or GPU memory measurement was performed.
