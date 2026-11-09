# CVE supervision cache

`prepare_cve.py` generates caliber supervision on CPU. It imports no model and uses no CUDA. It does not train, generate predictions, or overwrite a populated cache directory.

## Input and fixed splits

Use the project dataset layout:

- OCTA500-3M / OCTA500-6M: `train/image`, `train/label`, `val/image`, `val/label`, `test/image`, `test/label`.
- ROSSA: training combines `train_manual/image|label` and `train_sam/image|label`; validation/test use `val/image|label` and `test/image|label`.

Images and GT must be paired by identical filename. Membership must exactly match the project's `splits/<dataset>/{train,val,test}.txt`; no new split is sampled. Cache processing follows a local `random.Random(0)` permutation of each fixed list to preserve the validated processing order. It does not change global random state and is independent of the subsequent training seed.

GT is read in grayscale, converted to float32, resized to 224 x 224 with OpenCV nearest-neighbor interpolation, divided by 255, and thresholded at >0.5 only for caliber geometry.

## Algorithm and frozen parameters

1. Label 8-connected vessel components using a 3 x 3 all-ones structure; skeletonize the binary GT with scikit-image 0.21.0.
2. For every skeleton point, collect skeleton neighbors in the same component within radius 5 pixels. Require at least 3 points.
3. Compute the 2 x 2 covariance and eigenvectors using NumPy. The largest eigenvalue must exceed 1e-8; the ratio of largest to max(smallest,1e-8) must be at least 3.
4. Use the principal eigenvector as tangent and its perpendicular as normal. Sample both normal directions at step 0.25 pixels, from zero through the image diagonal plus one step.
5. Bilinearly sample the float32 component mask (`order=1`, `mode=constant`, `cval=0`, `prefilter=False`). Find the first value below boundary level 0.5 and interpolate its crossing within that step. Both boundaries must be observed inside the image; the resulting positive width must be finite.
6. Assign each vessel pixel the width of its nearest reliable skeleton point within the same component. Components without reliable points remain invalid with zero caliber. Background remains zero and invalid.
7. Pool the float32 widths at valid training skeleton samples only. Compute NumPy's default linear 99th percentile; validation and test never enter this fit. Do not estimate it from all vessel pixels, which would reweight wide vessels.
8. Normalize every split by the same dataset-specific training P99: `clip(raw / p99_train, 0, 1).astype(float32)`. Each dataset has its own P99. No fixed normalization value is embedded in the generator.

The geometry function preserves its validated operation ordering and dtypes. Covariance/eigenvectors, normal coordinates, and rays use the original float64 calculations; saved width maps and sample vectors are float32. A distance-transform map is not substituted for this definition.

## Commands

Run from the repository root after installing the pinned dependencies. `--workers` controls independent CPU sample processing; its default is 1.

```bash
python prepare_cve.py --dataset 3m   --data-root data/OCTA500_3M   --output-dir data/CVE_OCTA500_3M_maps --workers 4

python prepare_cve.py --dataset 6m   --data-root data/OCTA500_6M   --output-dir data/CVE_OCTA500_6M_maps --workers 4

python prepare_cve.py --dataset rossa   --data-root data/ROSSA   --output-dir data/CVE_ROSSA_maps --workers 4
```

Use a new or empty output directory. An output inside the input dataset or an existing completed cache is rejected. There is no overwrite option. A failed run retains its incomplete output and lock; choose a fresh directory after inspecting the failure. Completion is indicated by `prepared.json` and removal of `preparing.lock`.

## Output

```text
<output-dir>/
  prepared.json
  data_manifest.json
  caliber_quality.json
  caliber/
    train/
    val/
    test/
```

Each split contains four files per image:

| File | Shape | dtype | Purpose |
|---|---|---|---|
| `<stem>.raw.npy` | 224 x 224 | float32 | Unnormalized local caliber, in resized GT pixels |
| `<stem>.valid.npy` | 224 x 224 | bool | Reliable caliber-assignment region |
| `<stem>.samples.npy` | N | float32 | Reliable centerline width samples |
| `<stem>.caliber.npy` | 224 x 224 | float32 | Normalized caliber supervision |

`datasets/caliber.py` directly reads caliber and valid. The other arrays preserve normalization provenance. Metadata includes geometry, split counts, processing order, training-only P99, dependency versions, and an array-file digest. It does not need the research project's source code, run records, checkpoint, or private paths.

The V head is supervised directly with the segmentation GT already returned by the training Dataset (nearest-neighbor resize and float32/255). It is not derived from caliber, filtered by the caliber validity mask, or stored as a separate vesselness cache. The model's forward pass receives images and predicts R/V itself; this generator produces supervision targets.

## Training integration

```bash
python train.py --model m2rcnet --dataset 3m --seed 0   --data-root data/OCTA500_3M --map-root data/CVE_OCTA500_3M_maps
```

Use the corresponding dataset and cache roots for 6m or rossa. Configurations without CVE do not need these maps. Training loss, model construction, and checkpoint behavior are unchanged.

## Dependencies and source

The project caliber-geometry implementation is extracted from its validated research implementation; only IO and cache orchestration are separated. It calls public NumPy, SciPy, and scikit-image APIs; it does not copy those libraries' source. The fixed environment uses NumPy 1.24.4, SciPy 1.10.1, scikit-image 0.21.0, and opencv-python-headless 4.8.1.78. Their licenses remain with the respective projects. See the root LICENSE and README.md for the project license and attribution.

- [NumPy 1.24.4 license](https://github.com/numpy/numpy/blob/v1.24.4/LICENSE.txt)
- [SciPy 1.10.1 license](https://github.com/scipy/scipy/blob/v1.10.1/LICENSE.txt)
- [scikit-image 0.21.0 license](https://github.com/scikit-image/scikit-image/blob/v0.21.0/LICENSE.txt)
