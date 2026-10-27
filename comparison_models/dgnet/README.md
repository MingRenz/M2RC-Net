# DGNet integration

- Model: DGNet (Direction-guided network for retinal vessel segmentation in OCTA images).
- Official source path: /home/zzf/myeye/comparison_upstream/DGNet (read-only reference).
- Original repository: supplied DGNet-main.zip; no repository URL or revision is included in its README or Git metadata.
- Paper link supplied by upstream: https://www.sciencedirect.com/science/article/pii/S1746809424015131
- Model entry: DGNet.DGNet(inp_c=1, input_resolution=(224,224), n_classes=1); native layers=(32,32,32,32,32).
- Input: float32 [B,1,224,224], unchanged shared OCTA preprocessing.
- Output: native sigmoid probability [B,1,224,224], returned unchanged by the adapter.
- Parameters: 1,042,643 total/trainable.
- Required dependencies: torch only.
- Adaptation changes: two package-relative imports; fixed direction kernels become non-persistent module buffers rather than import-time CUDA globals; temporary features follow the input device/dtype.
- Architecture modifications: None. Direction kernels, attention, dropout, encoder/decoder and final sigmoid are retained. No trainable parameters are added by device adaptation.
- Training-specific notes: random initialization; shared comparison data/optimizer/Dice/Otsu/checkpoint pipeline is reused. No model-specific preprocessing is added.
- Auxiliary supervision: no auxiliary output is returned. The attribute named aux is the final segmentation head, not a discarded deep-supervision head.
- Known compatibility notes: supplied upstream contains model source and a short README, but no training/loss code. No extra core supervision is evidenced in this supplied implementation; this is not a claim that an unpublished official training objective was audited. LayerNorm uses configured square spatial dimensions, so the adapter fixes the documented 224x224 input.
- Smoke verification: passed synthetic [1,1,224,224] CPU forward and Dice backward with finite nonzero gradients. Same-seed initial state and fixed-input eval output are bitwise equal to the supplied upstream. No formal training or Test images.
