"""One real training batch plus full Val10 through each unified engine branch."""
import json
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine.runner import config_and_flags, build_training_objects, epoch_train, epoch_evaluate
from utils.deterministic import seed_all

if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG missing")
report={}
for variant in ("base","cams","mbrc","cams_mbrcnet"):
    config,_=config_and_flags(ROOT/"configs/octa500_3m.yaml",variant)
    seed_all(0)
    sets,loaders,model,weights,dice,optimizer=build_training_objects(
        config,variant,torch.device("cuda:0"))
    one=DataLoader(Subset(sets["train"],[0,1]),batch_size=2,shuffle=False,num_workers=0)
    train=epoch_train(variant,model,weights,dice,optimizer,one,torch.device("cuda:0"),1)
    val=epoch_evaluate(variant,model,weights,dice,loaders["val"],torch.device("cuda:0"),1,"VAL")
    report[variant]={"train":train,"val":val}
    print("ENGINE_SMOKE",variant,"PASS",flush=True)
    del sets,loaders,model,weights,dice,optimizer,one,train,val
    torch.cuda.empty_cache()
(ROOT/"tests/engine_smoke_report.json").write_text(json.dumps(report,indent=2))
