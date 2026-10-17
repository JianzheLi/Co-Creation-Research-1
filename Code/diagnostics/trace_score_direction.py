import json, math, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import configure,load_config,snapshots_for
from src.experiment import make_feedback
run=Path(r'../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve(); c=load_config(run/'config.yaml'); d=configure(c); truth=snapshots_for(c,full=True)
cp=torch.load(run/'feedback.pt',map_location=d,weights_only=False); m=make_feedback(c,d); m.load_state_dict(cp['model']);m.eval()
idx=torch.linspace(0,len(truth[0].x)-1,512,device=d).long(); x=torch.as_tensor(truth[0].x,dtype=torch.float32,device=d)[idx].clone(); gen=torch.Generator(device=d);gen.manual_seed(342); rows=[]; h=1/200; sig=c['model_sigma']
with torch.no_grad():
 for interval in range(2):
  for j in range(200):
   s=j*h; sn=(j+1)*h; u=math.sin(.5*math.pi*s)**2; un=math.sin(.5*math.pi*sn)**2; dt=un-u; dtds=.5*math.pi*math.sin(math.pi*s); t=torch.full((len(x),1),interval+u,dtype=x.dtype,device=d)
   vr,vs,g,ss=m.matching_fields(t,x)
   if interval==1 and 88<=j<=125:
    i=239; xx=x[i]; vv=vr[i]; sv=ss[i]; coss=lambda a,b:float((a@b/(a.norm()*b.norm()+1e-12)).item())
    rows.append({'step':j,'time':interval+u,'xnorm':float(xx.norm().item()),'xabs':float(xx.abs().max().item()),'scoreNorm':float(sv.norm().item()),'score_x_cos':coss(sv,xx),'vregNorm':float(vv.norm().item()),'vreg_x_cos':coss(vv,xx),'g':float(g[i].item()),'logw?':None})
   x=x+(dtds*vr+math.pi*sig*vs+.5*math.pi*sig*ss)*h+sig*torch.randn(x.shape,dtype=x.dtype,device=d,generator=gen)*math.sqrt(dt)
   if not torch.isfinite(x).all():break
  if not torch.isfinite(x).all():break
print(json.dumps(rows,indent=2)); out=run/'evaluation_all_initial_points'/'first_score_outlier_dynamics.json';out.write_text(json.dumps(rows,indent=2),encoding='utf-8');print('saved',out)
