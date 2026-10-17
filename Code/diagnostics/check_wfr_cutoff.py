from pathlib import Path
import numpy as np, json, math
root=Path.cwd(); cutoff=math.pi*math.sqrt(2)
for name in ['gene','mouse','emt']:
 run=root/'../Results'/f'{name}_distance_hermite_primal_alpha2_sigma1_seed42'
 rows=[]
 for f in sorted((run/'couplings').glob('interval*.npz')):
  with np.load(f) as z:
   x=z['source_positions']; y=z['target_positions']; p=z['P']
  d=np.linalg.norm(x[:,None,:]-y[None,:,:],axis=-1)
  total=p.sum(); outside=p[d>=cutoff].sum()
  rows.append({'file':f.name,'outside_cutoff_plan_mass':float(outside),'fraction':float(outside/total),'max_d_positive_P':float(d[p>0].max()) if np.any(p>0) else None})
 print(name,json.dumps(rows))
