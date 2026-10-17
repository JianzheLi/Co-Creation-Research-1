import json
from pathlib import Path
run=Path(r'../Results/experiment2/eb_noscale_silu_residual_w512x8_a1225_sigma05_blocks8_fm30000_seed42')
rows=[json.loads(s) for s in (run/'matching_history.jsonl').read_text().splitlines() if s.strip()]
for lo,hi in [(0,5000),(5000,10000),(10000,15000),(15000,20000),(20000,25000),(25000,30000)]:
 part=[r for r in rows if lo<r['step']<=hi]
 print(json.dumps({'range':[lo+1,hi],'count':len(part),'min_loss':min((r['loss'] for r in part),default=None),'last':part[-1] if part else None,'max_grad_norm':max((r['grad_norm'] for r in part),default=None)}))
print('first',rows[0]); print('last',rows[-1])
