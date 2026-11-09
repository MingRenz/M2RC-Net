import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from datasets import build_datasets,make_loaders
from models.base_mscf import build_base_mscf,EXPECTED_PARAMETERS
from losses.segmentation import DiceLoss
from utils.metrics import calc_result
from utils.checkpoint import save_json,save_weights,load_checkpoint
from utils.deterministic import seed_all
from utils.logger import append_history
from .runner import config_and_flags
from .trainer import train_base_epoch
from .validator import is_new_best,should_stop
from .m2rcnet_without_cve import state_hash,status

@torch.no_grad()
def evaluate_single(model,loader,criterion,device,label,complete=False,prediction_dir=None):
    model.eval();rows=[];losses=[]
    for names,image,gt in tqdm(loader,desc=label,total=len(loader),file=sys.stdout,ncols=110,mininterval=1.0):
        image=image.to(device,dtype=torch.float32);gt=gt.to(device,dtype=torch.float32)
        logits=model(image)
        assert torch.is_tensor(logits) and logits.shape==gt.shape
        probability=torch.sigmoid(logits)
        loss=criterion(probability,gt)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite evaluation Dice loss')
        losses.extend([float(loss)]*len(names))
        for j,name in enumerate(names):
            p=probability[j,0].cpu().numpy();y=gt[j,0].cpu().numpy()
            assert p.shape==y.shape==(224,224) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
            values=calc_result(p,y,thresh_value=None)
            row={'image_id':name,'dice':float(values['dice'])}
            if complete:
                row.update({'iou':float(values['iou']),'se':float(values['sen']),'sp':float(values['spe']),
                    'auc':float(roc_auc_score(y.reshape(-1),p.reshape(-1))),'acc':float(values['acc'])})
                if prediction_dir is not None:
                    import cv2
                    u8=cv2.normalize(p,None,0,255,cv2.NORM_MINMAX).astype('uint8')
                    threshold,mask=cv2.threshold(u8,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    np.savez_compressed(Path(prediction_dir)/(Path(name).stem+'.npz'),
                        probability=p,mask=mask.astype(np.uint8),gt=y,otsu_threshold=np.asarray(threshold))
            rows.append(row)
    assert len(rows)==len(loader.dataset) and len({r['image_id'] for r in rows})==len(rows)
    result={'loss':float(np.mean(losses)),'dice':float(np.mean([r['dice'] for r in rows])),'images':len(rows)}
    if complete:result.update({k:float(np.mean([r[k] for r in rows])) for k in ['iou','se','sp','auc','acc']})
    return result,rows


def train(config_path,run,*,config=None):
    run=Path(run).resolve()
    config,_=config_and_flags(config if config is not None else config_path,'mscf')
    seed=int(config['seed'])
    run.mkdir(parents=True,exist_ok=True)
    for name in ['metadata.json','history.jsonl','model_best.pth','finished.flag','test_result.json']:
        assert not (run/name).exists(),'Existing output; refusing overwrite: '+name
    seed_all(seed);device=torch.device('cuda:0')
    sets=build_datasets(config['dataset'],config['data_root'],use_cve=False,seed=seed)
    model=build_base_mscf().to(device)
    count=sum(p.numel() for p in model.parameters());assert count==EXPECTED_PARAMETERS
    assert not any('mbrc' in k or 'caliber_head' in k or 'vesselness_head' in k or k.rsplit('.',1)[-1] in ['a3','a5','a7'] for k in model.state_dict())
    criterion=DiceLoss();optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    loaders=make_loaders(sets,config['batch_size'],seed=seed)
    splits={s:d.names for s,d in sets.items()}
    metadata={'model':'BaseMSCF','model_class':type(model).__module__+'.'+type(model).__name__,
        'dataset':config['dataset'],'variant':'mscf','seed':seed,'data_root':str(Path(config['data_root']).resolve()),'map_root':None,
        'split':{s:len(d) for s,d in sets.items()},'resolution':[224,224],'batch_size':2,
        'optimizer':'Adam','learning_rate':1e-4,'weight_decay':1e-3,'max_epochs':400,'patience':40,
        'parameters':count,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'initialization':'random scratch, no existing/pretrained state assigned','initial_model_hash':state_hash(model),
        'initialization_comparison':{'performed_this_run':False,'result':None},
        'backbone_class_reused':'models.mscf.MSCFBackbone',
        'CVE':False,'MBRC':False,'loss':'DiceLoss(sigmoid(single segmentation logits), GT)',
        'loss_weights':None,'CaliberDataset':False,'segmentation_heads':1,'VSS_count':24,
        'selection':'strictly greater Val image-mean Otsu Dice','test_policy':'once after training, reload Val-best',
        'epoch_numbering':'1..400','augmentation':False,'scheduler':None,'AMP':False,
        'gradient_clipping':False,'gradient_accumulation':1,
        'train_loss_logging':'raw Dice loss. Existing train_base_epoch logs loss/batch_size; the returned value is multiplied by 2 for logging only. Backprop loss unchanged.',
        'strict_determinism':torch.are_deterministic_algorithms_enabled(),'config':str(Path(config_path).resolve()),
        'torch':torch.__version__,'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'python':sys.version}
    save_json(run/'metadata.json',metadata);save_json(run/'config.json',config);save_json(run/'split_manifest.json',splits)
    print('TRAINING_METADATA',json.dumps(metadata,ensure_ascii=False),flush=True)
    best=-1.0;best_epoch=0;last=0;checkpoint=run/'model_best.pth';started=time.time()
    for epoch in range(1,401):
        tick=time.time();status(run,'training',epoch=epoch,best_epoch=best_epoch,best_val_dice=best)
        logged=train_base_epoch(model,loaders['train'],criterion,optimizer,device,epoch)
        assert len(sets['train'])%2==0 and loaders['train'].batch_size==2
        train_loss=logged*2.0
        if not np.isfinite(train_loss):raise RuntimeError('Nonfinite training Dice loss')
        val,_=evaluate_single(model,loaders['val'],criterion,device,f'VAL {epoch}/400')
        improved=is_new_best(val['dice'],best)
        if improved:
            best=val['dice'];best_epoch=epoch
            save_weights(checkpoint,{'model_state_dict':model.state_dict(),'best_epoch':epoch,'best_val_dice':best,
                'variant':'mscf','seed':seed})
            save_json(run/'best_result.json',{'epoch':epoch,'val':val,'checkpoint':str(checkpoint)})
        record={'epoch':epoch,'train_loss':train_loss,'val_loss':val['loss'],'current_val_dice':val['dice'],
            'saved_best_epoch':best_epoch,'saved_best_val_dice':best,'saved_best_test_dice':None,
            'saved_this_epoch':improved,'epoch_seconds':time.time()-tick,'lr':optimizer.param_groups[0]['lr']}
        append_history(run,record)
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val['loss']:.6f} current_val_dice={val['dice']:.9f} "
              f"saved_best_epoch={best_epoch} saved_best_val_dice={best:.9f} saved_best_test_dice=NA "
              f"saved_this_epoch={improved} epoch_seconds={record['epoch_seconds']:.1f}",flush=True)
        last=epoch
        if should_stop(epoch,best_epoch,40):break
    status(run,'reload_best',best_epoch=best_epoch)
    loaded=load_checkpoint(checkpoint,model,map_location='cpu')
    assert loaded['variant']=='mscf' and loaded['best_epoch']==best_epoch and loaded['seed']==seed
    del loaded
    reloaded,_=evaluate_single(model,loaders['val'],criterion,device,'RELOAD VAL')
    assert reloaded['dice']==best,'Best Val Dice failed checkpoint reload verification'
    status(run,'testing',best_epoch=best_epoch)
    predictions=run/'predictions';predictions.mkdir()
    test,rows=evaluate_single(model,loaders['test'],criterion,device,'TEST BEST',complete=True,prediction_dir=predictions)
    assert len(rows)==len(sets['test']) and {r['image_id'] for r in rows}==set(splits['test'])
    with (run/'test_per_image.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_json(run/'test_result.json',{'checkpoint':str(checkpoint),'best_epoch':best_epoch,**test,
        'binary_protocol':'formal per-image minmax -> uint8 -> Otsu -> calc_result',
        'auc_protocol':'raw continuous probabilities, per-image ROC-AUC; arithmetic mean over test images'})
    summary={'model':'BaseMSCF','seed':seed,'parameters':count,'trained_epochs':last,'best_epoch':best_epoch,
        'best_val_dice':best,'test':test,'final_test_dice':test['dice'],'checkpoint':str(checkpoint),
        'stop_reason':'patience40' if should_stop(last,best_epoch,40) else 'max400',
        'elapsed_seconds':time.time()-started,'predictions':str(predictions),'success':True}
    save_json(run/'summary.json',summary)
    for name in ['model_best.pth','metadata.json','history.jsonl','summary.json','test_result.json','test_per_image.csv',
                 'console.log']:
        assert (run/name).is_file(),'Missing archive output: '+name
    status(run,'completed',exit_code=0,best_epoch=best_epoch,test_dice=test['dice'])
    (run/'finished.flag').write_text('training, best reload and test archive all succeeded\n',encoding='utf-8')
    print('FINAL_SUCCESS',json.dumps(summary,ensure_ascii=False),flush=True)
    return summary
