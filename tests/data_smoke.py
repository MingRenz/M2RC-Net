"""Split-order and real-batch equivalence, followed by one strict GPU step per dataset."""
import importlib.util
import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = Path("/home/zfzf/final_final/code")
NAMES = ("OCTA500_3M", "OCTA500_6M", "ROSSA")


def old_dataset_class(name):
    location = SOURCE / ("cams_mbrcnet_3m" if name == "OCTA500_3M" else "cams_mbrcnet_multidataset") / "dataset.py"
    spec = importlib.util.spec_from_file_location("old_data_"+name, location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OCTA3M if name == "OCTA500_3M" else module.OCTAMulti, module.seed_worker


def same_batch(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        if torch.is_tensor(x):
            assert torch.equal(x, y)
        else:
            assert list(x) == list(y)


def data_only():
    from datasets import build_dataset
    summary = {}
    for name in NAMES:
        old_class, worker = old_dataset_class(name)
        root = (ROOT / "data" / name).resolve()
        split_info = {}
        for split in ("train", "val", "test"):
            a = old_class(root, split)
            b = build_dataset(name, root, split)
            assert a.names == b.names
            assert len(a) == len(b)
            same_batch(a[0], b[0])
            split_info[split] = len(a)
        a = old_class(root, "train")
        b = build_dataset(name, root, "train")
        old_loader = DataLoader(a,batch_size=2,shuffle=True,drop_last=False,num_workers=3,
                                worker_init_fn=worker,generator=torch.Generator().manual_seed(0))
        new_loader = DataLoader(b,batch_size=2,shuffle=True,drop_last=False,num_workers=3,
                                worker_init_fn=worker,generator=torch.Generator().manual_seed(0))
        same_batch(next(iter(old_loader)), next(iter(new_loader)))
        summary[name] = split_info
        print("DATA_EQUAL",name,split_info,flush=True)
    return summary


def gpu_smoke():
    import os
    from models import build_model
    from losses.segmentation import DiceLoss
    from utils.deterministic import seed_all
    from datasets import build_dataloaders
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG missing")
    summary={}
    for name in NAMES:
        seed_all(0)
        _, loaders=build_dataloaders(name,ROOT/"data"/name,batch_size=2,seed=0)
        _names, image, gt=next(iter(loaders["train"]))
        model=build_model(False,False).cuda().train()
        output=model(image.cuda())
        loss=DiceLoss()(torch.sigmoid(output),gt.cuda())
        loss.backward()
        assert torch.isfinite(loss)
        assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.parameters())
        summary[name]={"batch_shape":list(image.shape),"loss":float(loss)}
        print("REAL_BATCH_SMOKE",name,summary[name],flush=True)
    return summary


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "data":
        report=data_only()
    else:
        report=gpu_smoke()
    Path("/tmp/cams_mbrcnet_"+sys.argv[1]+"_report.json").write_text(json.dumps(report,indent=2))
