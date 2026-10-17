import json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from scipy.spatial.distance import cdist
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code: src/
from src.evaluation_utils import emd2_with_retry as _emd2_with_retry
from src.runtime import configure, load_config, snapshots_for
from src.experiment import make_feedback
from src.sample import rollout
run=Path(r'../Results/experiment2/eb_noscale_leakyrelu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve()
cfg=load_config(run/'config.yaml'); dev=configure(cfg); truth=snapshots_for(cfg,full=True)
ck=torch.load(run/'feedback.pt',map_location=dev,weights_only=False); model=make_feedback(cfg,dev); model.load_state_dict(ck['model']); model.eval()
start=time.perf_counter(); out={'method':'LeakyReLU+residual','steps_per_unit':200,'full_n':len(truth[0].x)}
try:
 res=rollout(model,truth[0].x,cfg['model_times'],cfg['model_sigma'],len(truth[0].x),200,seed=342,endpoint_guard=cfg['generation']['endpoint_guard'],initial_x=truth[0].x,stochastic=True)
 p=res['positions'].astype(np.float64); w=np.exp(res['log_weights']); metrics=[]
 for k,snap in enumerate(truth):
  a=w[k]/w[k].sum(); b=np.full(len(snap.x),1/len(snap.x)); c=cdist(p[k],snap.x,'euclidean')
  w1=float(_emd2_with_retry(a,b,c)); w2=float(np.sqrt(_emd2_with_retry(a,b,c*c)))
  mr=float(w[k].sum()/w[0].sum()); obs=float(snap.mass/truth[0].mass)
  metrics.append({'time':float(snap.time),'W1':w1,'W2':w2,'predicted_mass_ratio':mr,'observed_mass_ratio':obs,'mass_relative_error':abs(mr-obs)/max(abs(obs),1e-12),'ESS':float(w[k].sum()**2/(w[k]@w[k])),'max_abs_position':float(np.abs(p[k]).max())})
 out.update(status='completed',metrics=metrics,elapsed_seconds=time.perf_counter()-start)
except Exception as e: out.update(status='failed',error=repr(e),elapsed_seconds=time.perf_counter()-start)
print(json.dumps(out,indent=2),flush=True)
(run/'evaluation_all_initial_points'/'sde_full_leaky_seed342.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
# Test stochastic robustness on a fixed 512-point evenly spaced subset over several independent noise seeds.
idx=torch.linspace(0,len(truth[0].x)-1,512,device=dev).long().cpu().numpy(); x=truth[0].x[idx]; robust=[]
for seed in range(340,345):
 try:
  z=rollout(model,torch.as_tensor(x,dtype=torch.float32,device=dev),cfg['model_times'],cfg['model_sigma'],len(x),200,seed=seed,endpoint_guard=cfg['generation']['endpoint_guard'],initial_x=torch.as_tensor(x,dtype=torch.float32,device=dev),stochastic=True)
  robust.append({'seed':seed,'status':'completed','max_abs_position_by_time':np.max(np.abs(z['positions']),axis=(1,2)).tolist()})
 except Exception as e: robust.append({'seed':seed,'status':'failed','error':repr(e)})
print(json.dumps({'subset_robustness':robust},indent=2),flush=True)
(run/'evaluation_all_initial_points'/'sde_subset_robustness_leaky.json').write_text(json.dumps(robust,indent=2),encoding='utf-8')
