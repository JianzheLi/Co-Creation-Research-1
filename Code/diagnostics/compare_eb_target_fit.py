import json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import configure, load_config, load_path
from src.matching import CouplingSampler, sample_training_batch, make_feedback, matching_loss_weights
base=Path(r'../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve()
cand=Path(r'../Results/experiment2/eb_noscale_leakyrelu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve()
cfg0=load_config(base/'config.yaml'); cfg1=load_config(cand/'config.yaml'); dev=configure(cfg0)
sampler=CouplingSampler(base,cfg0,dev); path=load_path(base,cfg0,dev)
models=[]
for run,cfg in [(base,cfg0),(cand,cfg1)]:
 ck=torch.load(run/'feedback.pt',map_location=dev,weights_only=False); m=make_feedback(cfg,dev); m.load_state_dict(ck['model']); m.eval(); models.append(m)
gen=torch.Generator(device=dev); gen.manual_seed(9472)
keys=['vreg','vsing','score','growth']; store={tag:{name:{scope:{f:[] for f in ['target','pred','err']} for scope in ['all','i1late']} for name in keys} for tag in ['silu','leakyrelu']}
with torch.no_grad():
 for b in range(16):
  t,tar,fac=sample_training_batch(sampler,path,2048,dev,generator=gen)
  local=t-t.floor(); mask=(t.floor().squeeze(-1)==1)&(local.squeeze(-1)>=.75)&(local.squeeze(-1)<.98)
  targets={'vreg':tar['target_v_regular'].norm(dim=-1),'vsing':tar['target_v_singular_scaled'].norm(dim=-1),'score':tar['target_s_scaled'].norm(dim=-1),'growth':tar['target_g'].abs().squeeze(-1)}
  for tag,m in zip(['silu','leakyrelu'],models):
   pred=m.matching_fields(t,tar['x']); pp={'vreg':pred[0].norm(dim=-1),'vsing':pred[1].norm(dim=-1),'score':pred[3].norm(dim=-1),'growth':pred[2].abs().squeeze(-1)}
   for field in keys:
    err=(pred[0]-tar['target_v_regular']).norm(dim=-1) if field=='vreg' else (pred[1]-tar['target_v_singular_scaled']).norm(dim=-1) if field=='vsing' else (pred[3]-tar['target_s_scaled']).norm(dim=-1) if field=='score' else (pred[2]-tar['target_g']).abs().squeeze(-1)
    for scope,maskval in [('all',None),('i1late',mask)]:
     store[tag][field][scope]['target'].append(targets[field].detach() if maskval is None else targets[field][maskval].detach())
     store[tag][field][scope]['pred'].append(pp[field].detach() if maskval is None else pp[field][maskval].detach())
     store[tag][field][scope]['err'].append(err.detach() if maskval is None else err[maskval].detach())
def summ(vals):
 a=torch.cat(vals).double(); return {'n':int(a.numel()),'rms':float(a.square().mean().sqrt().item()),'p99':float(torch.quantile(a,.99).item()),'p999':float(torch.quantile(a,.999).item()),'max':float(a.max().item())}
out={}
for tag in store:
 out[tag]={}
 for f in keys:
  out[tag][f]={}
  for scope in ['all','i1late']:
   z=store[tag][f][scope]; s={k:summ(v) for k,v in z.items()}; s['relative_rms_error']=s['err']['rms']/max(s['target']['rms'],1e-12); out[tag][f][scope]=s
out['weights']={'silu':matching_loss_weights(cfg0),'leakyrelu':matching_loss_weights(cfg1)}
out['target_sampler_seed']=9472
outfile=cand/'matching_target_fit_comparison.json'; outfile.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2)); print('saved',outfile)
