import json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from src.runtime import configure, load_config, snapshots_for
from src.experiment import make_feedback
from src.sample import rollout
run=Path(r'../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve()
cfg=load_config(run/'config.yaml'); dev=configure(cfg)
ck=torch.load(run/'feedback.pt',map_location=dev,weights_only=False); model=make_feedback(cfg,dev); model.load_state_dict(ck['model']); model.eval()
truth=snapshots_for(cfg,full=True); idx=torch.linspace(0,len(truth[0].x)-1,512,device=dev).long().cpu().numpy(); x=truth[0].x[idx]
rows=[]
for nsteps in [200,400,800]:
 for seed in [340,341,342]:
  try:
   res=rollout(model,torch.as_tensor(x,dtype=torch.float32,device=dev),cfg['model_times'],cfg['model_sigma'],len(x),nsteps,seed=seed,endpoint_guard=cfg['generation']['endpoint_guard'],initial_x=torch.as_tensor(x,dtype=torch.float32,device=dev),stochastic=True)
   pos=res['positions']; lw=res['log_weights']; row={'steps':nsteps,'seed':seed,'status':'completed','max_abs_position':np.max(np.abs(pos),axis=(1,2)).tolist(),'max_abs_log_weight':np.max(np.abs(lw),axis=1).tolist()}
  except Exception as e: row={'steps':nsteps,'seed':seed,'status':'failed','error':repr(e)}
  rows.append(row); print(json.dumps(row),flush=True)
out=run/'evaluation_all_initial_points'/'sde_step_size_diagnostic.json'; out.write_text(json.dumps(rows,indent=2),encoding='utf-8'); print('saved',out)
