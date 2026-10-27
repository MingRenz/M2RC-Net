# PMDC-Net integration

- Model: PMDC-Net.
- Official source path: /home/zzf/myeye/comparison_upstream/PMDC-Net (read-only reference).
- Original repository: https://github.com/fdding/PMDC-Net (from supplied README); supplied archive has no Git revision metadata.
- Model entry: src/model/pmdc_net.py, PMDC_Net(n_channels=1,n_classes=2,bilinear=False,use_msfam=True,use_cwm=True,use_pdcm=True,use_dafm=True).
- Input: float32 [B,1,224,224], unchanged shared OCTA preprocessing.
- Original output: one [B,2,224,224] raw-logit tensor.
- Adapter output: torch.softmax(raw_logits,dim=1)[:,1:2], a [B,1,224,224] foreground probability. No sigmoid, output-head replacement, interpolation or threshold is added.
- Parameters: 17,432,342 total/trainable.
- Required dependencies: torch and Python standard library math only.
- Adaptation changes: a separate explicit two-class-to-binary-probability adapter. pmdc_net.py is copied unchanged.
- Architecture modifications: None. All MSFAM, CWM, PDCM and DAFM switches remain enabled, the two-class head and native transposed-convolution path are retained.
- Training-specific notes: upstream src/utils/loss.py implements 0.3 CE + 0.7 multiclass Dice, including both background and foreground. This comparison uses the existing foreground Dice-only common protocol; it is not an official-training-recipe reproduction. The supplied source does not include a training entry, so actual official optimizer/schedule/constructor selections cannot be confirmed solely from code; README-reported 512 input and other recipe settings are not silently adopted.
- Auxiliary supervision: no auxiliary outputs, deep-supervision heads or method-specific extra targets were found in the supplied model.
- Known compatibility notes: bilinear=False is kept. The alternate bilinear=True channel path has not been validated. No weights, CUDA device, data paths or downloads are hardcoded by this model.
- Smoke verification: passed synthetic [1,1,224,224] CPU forward and Dice backward, including nonzero gradients in both class logits. Same-seed initial state and fixed-input raw logits are bitwise equal to supplied upstream. No formal training or Test images.
