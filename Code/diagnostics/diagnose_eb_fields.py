import json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.runtime import configure, load_config, snapshots_for
from src.experiment import make_feedback

run = Path(r'../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve()
config = load_config(run/'config.yaml')
device = configure(config)
ckpt = torch.load(run/'feedback.pt', map_location=device, weights_only=False)
model = make_feedback(config, device)
model.load_state_dict(ckpt['model']); model.eval()
truth = snapshots_for(config, full=True)
idx = torch.linspace(0, len(truth[0].x)-1, steps=512, device=device).long()
x = torch.as_tensor(truth[0].x, dtype=torch.float32, device=device)[idx].clone()
logw = torch.full((len(x),), -math.log(len(x)), dtype=torch.float64, device=device)
gen = torch.Generator(device=device); gen.manual_seed(342)
steps=200; ds=1/steps; sigma=float(config['model_sigma']); rows=[]
with torch.no_grad():
  for interval in range(len(config['model_times'])-1):
    for j in range(steps):
      s=j*ds; sn=(j+1)*ds
      local=math.sin(0.5*math.pi*s)**2
      localn=math.sin(0.5*math.pi*sn)**2
      dt=localn-local
      dtds=0.5*math.pi*math.sin(math.pi*s)
      t=torch.full((len(x),1), interval+local, dtype=x.dtype, device=device)
      vr,vs,g,ss=model.matching_fields(t,x)
      term_reg=dtds*vr
      term_sing=math.pi*sigma*vs
      term_score=0.5*math.pi*sigma*ss
      drift_s=term_reg+term_sing+term_score
      brownian=torch.randn(x.shape,dtype=x.dtype,device=device,generator=gen)*math.sqrt(dt)
      x=x+drift_s*ds+sigma*brownian
      logw=logw+(dtds*g.squeeze(-1).double())*ds
      if j % 5==0 or j==steps-1:
        def stat(v):
          n=v.reshape(len(v),-1).norm(dim=1)
          return {'max':float(n.max().item()),'p99':float(torch.quantile(n,0.99).item()),'mean':float(n.mean().item())}
        rows.append({'interval':interval, 'step':j+1, 'time':interval+localn,
          'x_abs_max':float(x.abs().max().item()),'x_abs_p999':float(torch.quantile(x.abs().reshape(-1),.999).item()),
          'v_regular':stat(term_reg),'v_singular':stat(term_sing),'score_drift':stat(term_score),
          'growth_abs_max':float(g.abs().max().item()),'growth_abs_p99':float(torch.quantile(g.abs().reshape(-1),.99).item()),
          'logw_abs_max':float(logw.abs().max().item()), 'finite':bool(torch.isfinite(x).all() and torch.isfinite(logw).all())})
      if not torch.isfinite(x).all() or not torch.isfinite(logw).all(): break
    if not torch.isfinite(x).all() or not torch.isfinite(logw).all(): break
out={'dataset':config['dataset'],'n':len(x),'steps_per_interval':steps,'seed':342,'rows':rows}
path=run/'evaluation_all_initial_points'/'field_diagnostic_seed342.json'
path.write_text(json.dumps(out,indent=2),encoding='utf-8')
for r in rows:
  if r['interval']>=1 and (r['step']%20==1 or not r['finite'] or r['step']==200): print(json.dumps(r))
print('saved',path)
