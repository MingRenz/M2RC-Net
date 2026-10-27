"""One in-memory optimizer step with real Train data and precomputed CAMS maps."""
import json
import os
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine.runner import config_and_flags, build_training_objects
from engine.trainer import joint_calculate
from utils.deterministic import seed_all

if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG missing")
report={}
for name in ("octa500_3m","octa500_6m","rossa"):
    config,_=config_and_flags(ROOT/"configs"/(name+".yaml"),"cams_mbrcnet")
    seed_all(0)
    datasets,loaders,model,weights,dice,optimizer=build_training_objects(
        config,"cams_mbrcnet",torch.device("cuda:0"))
    batch=next(iter(loaders["train"]))
    optimizer.zero_grad(set_to_none=True)
    _names,out,gt,parts=joint_calculate(model,weights,dice,batch,torch.device("cuda:0"))
    parts["total"].backward()
    assert bool(torch.isfinite(parts["total"]))
    assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.parameters())
    optimizer.step();weights.project()
    report[name]={"count":len(datasets["train"]),"shape":list(batch[1].shape),
                  "caliber_shape":list(batch[3].shape),"loss":float(parts["total"])}
    print("FULL_REAL_BATCH_SMOKE",name,report[name],flush=True)
    del datasets,loaders,model,weights,dice,optimizer,batch,out,gt,parts
    torch.cuda.empty_cache()
Path("/tmp/cams_mbrcnet_full_real_smoke.json").write_text(json.dumps(report,indent=2))
