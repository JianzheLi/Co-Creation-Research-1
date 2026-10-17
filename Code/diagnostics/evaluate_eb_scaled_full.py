import json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from scipy.spatial.distance import cdist
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DeepRUOT/Code: src/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # workspace root: Baselines/
from Baselines.experiment2.evaluate import _emd2_with_retry
from src.runtime import configure, load_config, snapshots_for
from src.experiment import make_feedback
from src.sample import rollout
run=Path(r'../Results/experiment2/eb_noscale_silu_depthscaled_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve(); outdir=run/'evaluation_all_initial_points'; outdir.mkdir(parents=True,exist_ok=True)
cfg=load_config(run/'config.yaml');dev=configure(cfg);truth=snapshots_for(cfg,full=True);cp=torch.load(run/'feedback.pt',map_location=dev,weights_only=False);model=make_feedback(cfg,dev);model.load_state_dict(cp['model']);model.eval();start=time.perf_counter()
try:
 z=rollout(model,truth[0].x,cfg['model_times'],cfg['model_sigma'],len(truth[0].x),200,seed=342,endpoint_guard=cfg['generation']['endpoint_guard'],initial_x=truth[0].x,stochastic=True)
 p=z['positions'].astype(np.float64);w=np.exp(z['log_weights']); metrics=[]
 for k,snap in enumerate(truth):
  a=w[k]/w[k].sum();b=np.full(len(snap.x),1/len(snap.x));c=cdist(p[k],snap.x,'euclidean');w1=float(_emd2_with_retry(a,b,c));w2=float(np.sqrt(_emd2_with_retry(a,b,c*c)));mr=float(w[k].sum()/w[0].sum());obs=float(snap.mass/truth[0].mass)
  metrics.append({'time':float(snap.time),'W1':w1,'W2':w2,'predicted_mass_ratio':mr,'observed_mass_ratio':obs,'mass_relative_error':abs(mr-obs)/max(abs(obs),1e-12),'ESS':float(w[k].sum()**2/(w[k]@w[k])),'max_abs_position':float(np.abs(p[k]).max())})
 result={'status':'completed','inference':'SDE','source_points':len(truth[0].x),'sigma':cfg['model_sigma'],'steps_per_unit':200,'seed':342,'metrics':metrics,'elapsed_seconds':time.perf_counter()-start};np.savez_compressed(outdir/'sde_predictions_seed342.npz',times=z['times'],positions=p,weights=w,seed=342,sigma=cfg['model_sigma'])
except Exception as e: result={'status':'failed','inference':'SDE','source_points':len(truth[0].x),'sigma':cfg['model_sigma'],'steps_per_unit':200,'seed':342,'error':repr(e),'elapsed_seconds':time.perf_counter()-start}
(outdir/'sde_full_metrics_seed342.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
