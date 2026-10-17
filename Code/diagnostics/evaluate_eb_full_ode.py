import json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DeepRUOT/Code: src/
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # workspace root: Baselines/
from Baselines.experiment2.evaluate import _emd2_with_retry
from src.runtime import configure, load_config, snapshots_for, file_hash
from src.experiment import make_feedback
from src.sample import rollout
run=Path(r'../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve()
cfg=load_config(run/'config.yaml'); dev=configure(cfg); truth=snapshots_for(cfg,full=True)
ck=torch.load(run/'feedback.pt',map_location=dev,weights_only=False); model=make_feedback(cfg,dev); model.load_state_dict(ck['model']); model.eval()
start=time.perf_counter()
res=rollout(model,truth[0].x,cfg['model_times'],cfg['model_sigma'],len(truth[0].x),200,seed=342,endpoint_guard=cfg['generation']['endpoint_guard'],initial_x=truth[0].x,stochastic=False)
rows=[]; w=np.exp(res['log_weights']); p=res['positions'].astype(np.float64)
for k,snap in enumerate(truth):
 mass=w[k]; a=mass/mass.sum(); b=np.full(len(snap.x),1/len(snap.x)); c=cdist(p[k],snap.x,'euclidean')
 w1=float(_emd2_with_retry(a,b,c)); w2=float(np.sqrt(_emd2_with_retry(a,b,c*c)))
 observed=snap.mass/truth[0].mass; pred=float(mass.sum()/w[0].sum())
 ess=float(mass.sum()**2/(mass**2).sum())
 rows.append({'time':float(snap.time),'W1':w1,'W2':w2,'predicted_mass_ratio':pred,'observed_mass_ratio':observed,'mass_relative_error':abs(pred-observed)/max(abs(observed),1e-12),'ESS':ess,'max_abs_position':float(np.abs(p[k]).max()),'max_log_weight':float(np.max(res['log_weights'][k]))})
 out=run/'evaluation_all_initial_points'/'ode_full_metrics_seed342.json'; out.write_text(json.dumps({'inference':'ODE','initial_points':len(truth[0].x),'sigma':cfg['model_sigma'],'steps_per_unit':200,'seed':342,'elapsed_seconds':time.perf_counter()-start,'metrics':rows},indent=2),encoding='utf-8')
print(json.dumps(rows,indent=2)); print('saved',out)
