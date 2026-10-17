import json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.runtime import configure,load_config,load_path,snapshots_for
from src.matching import CouplingSampler,sample_training_batch
from src.experiment import make_feedback
base=Path(r'../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42').resolve();c=load_config(base/'config.yaml');d=configure(c);truth=snapshots_for(c,full=True)
cp=torch.load(base/'feedback.pt',map_location=d,weights_only=False);m=make_feedback(c,d);m.load_state_dict(cp['model']);m.eval()
idx=torch.linspace(0,len(truth[0].x)-1,512,device=d).long();x=torch.as_tensor(truth[0].x,dtype=torch.float32,device=d)[idx].clone();gen=torch.Generator(device=d);gen.manual_seed(342);h=1/200;sig=c['model_sigma']; points={}
with torch.no_grad():
 for k in range(2):
  for j in range(130):
   s=j*h;sn=(j+1)*h;u=math.sin(.5*math.pi*s)**2;un=math.sin(.5*math.pi*sn)**2;dt=un-u;dtds=.5*math.pi*math.sin(math.pi*s);t=torch.full((len(x),1),k+u,dtype=x.dtype,device=d);vr,vs,g,ss=m.matching_fields(t,x)
   if k==1 and j in [88,92,100,110,120,125]:points[str(j)]={'t':k+u,'x':x[239].clone(),'fields':[v[239].clone() for v in [vr,vs,g,ss]]}
   x=x+(dtds*vr+math.pi*sig*vs+.5*math.pi*sig*ss)*h+sig*torch.randn(x.shape,dtype=x.dtype,device=d,generator=gen)*math.sqrt(dt)
# estimate the largest local expansion rate of the transformed SDE drift using JVP/VJP power iteration
rows=[]
for key,p in points.items():
 xx=p['x'].detach().requires_grad_(True); tt=float(p['t']); local=tt-math.floor(tt);s=(2/math.pi)*math.asin(math.sqrt(local));dtds=.5*math.pi*math.sin(math.pi*s);t=torch.tensor([[tt]],dtype=xx.dtype,device=d)
 def f(z):
  vr,vs,g,ss=m.matching_fields(t,z[None,:]); return dtds*vr[0]+math.pi*sig*vs[0]+.5*math.pi*sig*ss[0]
 J=torch.autograd.functional.jacobian(f,xx,vectorize=True)
 sym=(J+J.T)*.5; eig=torch.linalg.eigvalsh(sym)
 rows.append({'step':int(key),'time':tt,'xnorm':float(xx.norm().item()),'score_norm':float(p['fields'][3].norm().item()),'vreg_norm':float(p['fields'][0].norm().item()),'growth':float(p['fields'][2].item()),'jacobian_spectral_norm':float(torch.linalg.matrix_norm(J,ord=2).item()),'symmetric_jacobian_max_eigenvalue':float(eig.max().item()),'symmetric_jacobian_min_eigenvalue':float(eig.min().item())})
print(json.dumps(rows,indent=2));out=base/'evaluation_all_initial_points'/'sde_local_jacobian_diagnostic.json';out.write_text(json.dumps(rows,indent=2),encoding='utf-8');print('saved',out)
