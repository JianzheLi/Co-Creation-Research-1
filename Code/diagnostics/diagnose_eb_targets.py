import json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from src.runtime import configure, load_config, load_path
from src.matching import CouplingSampler, sample_training_batch, make_feedback, matching_loss_weights
run=Path(r'../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve()
cfg=load_config(run/'config.yaml'); dev=configure(cfg); sampler=CouplingSampler(run,cfg,dev); path=load_path(run,cfg,dev)
ck=torch.load(run/'feedback.pt',map_location=dev,weights_only=False); model=make_feedback(cfg,dev); model.load_state_dict(ck['model']); model.eval(); weights=matching_loss_weights(cfg)
store={key:{'all':[],'i1late':[]} for key in ['xnorm','vreg_target','vreg_pred','vreg_err','vsing_target','vsing_pred','vsing_err','score_target','score_pred','score_err','growth_target','growth_pred','growth_err']}
with torch.no_grad():
 for b in range(12):
  t, tar, fac=sample_training_batch(sampler,path,2048,dev)
  pred=model.matching_fields(t,tar['x'])
  vreg,vsing,g,score=pred
  local=t-t.floor()
  mask=(t.floor().squeeze(-1)==1)&(local.squeeze(-1)>=.75)&(local.squeeze(-1)<.98)
  values={
   'xnorm':tar['x'].norm(dim=-1),
   'vreg_target':tar['target_v_regular'].norm(dim=-1),'vreg_pred':vreg.norm(dim=-1),'vreg_err':(vreg-tar['target_v_regular']).norm(dim=-1),
   'vsing_target':tar['target_v_singular_scaled'].norm(dim=-1),'vsing_pred':vsing.norm(dim=-1),'vsing_err':(vsing-tar['target_v_singular_scaled']).norm(dim=-1),
   'score_target':tar['target_s_scaled'].norm(dim=-1),'score_pred':score.norm(dim=-1),'score_err':(score-tar['target_s_scaled']).norm(dim=-1),
   'growth_target':tar['target_g'].abs().squeeze(-1),'growth_pred':g.abs().squeeze(-1),'growth_err':(g-tar['target_g']).abs().squeeze(-1)}
  for key,val in values.items():
   store[key]['all'].append(val.detach())
   store[key]['i1late'].append(val[mask].detach())
def summ(xs):
 if not xs: return None
 a=torch.cat(xs).double(); return {'n':int(a.numel()),'rms':float(a.square().mean().sqrt().item()),'p50':float(torch.quantile(a,.5).item()),'p95':float(torch.quantile(a,.95).item()),'p99':float(torch.quantile(a,.99).item()),'p999':float(torch.quantile(a,.999).item()),'max':float(a.max().item())}
out={scope:{key:summ(parts[scope]) for key,parts in store.items()} for scope in ['all','i1late']}
for scope in out:
 for field in ['vreg','vsing','score','growth']:
  k=f'{field}_'
  target=out[scope][k+'target']; err=out[scope][k+'err']
  if target and err: out[scope][field+'_relative_rms_error']=err['rms']/max(target['rms'],1e-12)
print(json.dumps(out,indent=2))
outfile=run/'evaluation_all_initial_points'/'matching_target_fit_diagnostic.json'; outfile.write_text(json.dumps(out,indent=2),encoding='utf-8'); print('saved',outfile)
