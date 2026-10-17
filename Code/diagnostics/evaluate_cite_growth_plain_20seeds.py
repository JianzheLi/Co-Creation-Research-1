
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code: src/
import json,math,sys,time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0,str(Path.cwd()))
from src.runtime import load_config,configure,snapshots_for,file_hash
from src.experiment import make_feedback
from src.matching import validate_feedback_architecture
from src.sample import rollout
run=Path('../Results/experiment2/cite_full_a1800_leakyrelu_prenorm_velocity_score_growth_plain_w512x8_growth4_lr3e3to1e5_b4096_s1500_blocks12_proj5e5_fm30000_seed42')
cfg=load_config(run/'config.yaml'); dev=configure(cfg); truth=snapshots_for(cfg,full=True)
cp=torch.load(run/'feedback.pt',map_location=dev,weights_only=False); validate_feedback_architecture(cp,cfg); model=make_feedback(cfg,dev); model.load_state_dict(cp['model']); model.eval()
outfile=run/'evaluation_all_initial_points'/'sde_full_20seeds_stability.json'
res={'dataset':'cite','sigma':cfg['model_sigma'],'alpha':cfg['growth_penalty'],'steps_per_unit':200,'source_points':len(truth[0].x),'seeds':[],'objective_unchanged':True,'checkpoint_sha256':file_hash(run/'feedback.pt')}; start=time.perf_counter()
for seed in range(342,362):
 t0=time.perf_counter()
 try:
  out=rollout(model,truth[0].x,times=cfg['model_times'],sigma=cfg['model_sigma'],n_particles=len(truth[0].x),steps_per_unit=200,seed=seed,endpoint_guard=cfg['generation']['endpoint_guard'],initial_x=truth[0].x,stochastic=True)
  ms=[]
  for k,snap in enumerate(truth):
   x=np.asarray(out['positions'][k]); lw=np.asarray(out['log_weights'][k],dtype=np.float64)
   if not np.isfinite(x).all() or not np.isfinite(lw).all(): raise FloatingPointError(f'nonfinite at t={snap.time}')
   z=float(lw.max()); weights=np.exp(lw-z); ess=float(weights.sum()**2/(weights@weights)); logmass=z+math.log(float(weights.sum())); mass=math.exp(logmass) if logmass<700 else float('inf')
   ms.append({'time':float(snap.time),'predicted_mass':mass,'observed_mass':float(snap.mass),'mass_relative_error':abs(mass-snap.mass)/max(snap.mass,1e-300),'ESS':ess,'max_abs_position':float(np.abs(x).max()),'max_abs_log_weight':float(np.abs(lw).max())})
  record={'seed':seed,'status':'completed','metrics':ms,'elapsed_seconds':time.perf_counter()-t0}
 except Exception as e: record={'seed':seed,'status':'failed','error':repr(e),'elapsed_seconds':time.perf_counter()-t0}
 res['seeds'].append(record)
 done=[r for r in res['seeds'] if r['status']=='completed']
 res['summary']={'completed_seeds':len(done),'failed_seeds':len(res['seeds'])-len(done),'max_abs_position_all_seeds':max((m['max_abs_position'] for r in done for m in r['metrics']),default=None),'max_abs_log_weight_all_seeds':max((m['max_abs_log_weight'] for r in done for m in r['metrics']),default=None),'mean_mass_relative_error_by_time':{str(int(s.time)):float(np.mean([next(m for m in r['metrics'] if m['time']==s.time)['mass_relative_error'] for r in done])) for s in truth[1:]}}
 res['elapsed_seconds']=time.perf_counter()-start; outfile.parent.mkdir(parents=True,exist_ok=True); outfile.write_text(json.dumps(res,indent=2),encoding='utf-8')
 print(json.dumps({'seed':seed,'status':record['status'],'metrics':record.get('metrics',[])[-1:] if record['status']=='completed' else record.get('error')},ensure_ascii=False),flush=True)
print('SUMMARY',json.dumps(res['summary'],ensure_ascii=False),flush=True)
