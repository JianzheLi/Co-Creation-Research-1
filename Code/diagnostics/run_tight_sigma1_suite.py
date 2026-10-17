"""Run the three requested complete pipelines, then matched MC comparisons."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '../Results/tight_bound_alpha2_sigma1_comparison'
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ['PYTHONUNBUFFERED'] = '1'


def command(args, logname):
    start = time.time()
    with (OUT / logname).open('w', encoding='utf8') as log:
        p = subprocess.Popen([sys.executable, '-u', *args], cwd=ROOT,
                             stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps(dict(event='started', pid=p.pid, args=args)), flush=True)
        code = p.wait()
    result = dict(args=args, returncode=code, seconds=time.time()-start, log=logname)
    print(json.dumps(result), flush=True)
    return result


def train(dataset):
    return dataset, command(['scripts/run_experiment.py', '--config',
        f'configs/{dataset}_tight_bound_alpha2_sigma1.yaml', '--run-dir',
        f'../Results/{dataset}_tight_bound_alpha2_sigma1_seed42'], dataset+'_pipeline.log')


def main():
    records = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(train, d) for d in ['gene','mouse','emt']]
        for future in as_completed(futures):
            dataset, result = future.result()
            records[dataset] = result
            (OUT/'suite_status.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    for dataset in ['gene','mouse','emt']:
        if records[dataset]['returncode']:
            continue
        for method, stem in [('tight','tight_bound'),('original','distance_hermite_primal')]:
            result = command(['scripts/compute_action_monte_carlo.py', '--ruot-run',
                f'../Results/{dataset}_{stem}_alpha2_sigma1_seed42', '--wfr-run',
                f'../Results/{dataset}_wfr_bridge_alpha2_sigma1_seed42', '--repeats','20',
                '--seed-start','1000','--out-dir',
                f'../Results/action_monte_carlo_{dataset}_{method}_alpha2_sigma1_seed1000'],
                f'{dataset}_{method}_mc.log')
            records[dataset][method+'_mc'] = result
            (OUT/'suite_status.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    print('Suite finished', flush=True)


if __name__ == '__main__':
    main()
