"""Check that two completed methods use identical observed blocks and weights."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(first, second):
    first, second = Path(first), Path(second)
    manifests = [json.loads((run/'couplings'/'manifest.json').read_text(encoding='utf-8'))
                 for run in (first, second)]
    for run, manifest in zip((first, second), manifests):
        status = json.loads((run/'status.json').read_text(encoding='utf-8'))
        assert status['couplings']['status'] == 'completed', run
        assert manifest['status'] in ('converged', 'audited'), run
    blocks = [{(row['interval'], row['block']): row for row in manifest['intervals']}
              for manifest in manifests]
    assert blocks[0].keys() == blocks[1].keys()
    fields = ['source_indices', 'target_indices', 'source_weights', 'target_weights',
              'source_positions', 'target_positions']
    for key in blocks[0]:
        with np.load(first/'couplings'/blocks[0][key]['file']) as a, \
             np.load(second/'couplings'/blocks[1][key]['file']) as b:
            for field in fields:
                assert np.array_equal(a[field], b[field]), (key, field)
    report = dict(passed=True, first_run=str(first.resolve()), second_run=str(second.resolve()),
                  compared_blocks=len(blocks[0]), identical_fields=fields,
                  manifest_statuses=[m['status'] for m in manifests],
                  block_solver_statuses=[[row['status'] for row in m['intervals']]
                                         for m in manifests])
    (first/'partition_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-run', required=True)
    parser.add_argument('--second-run', required=True)
    args = parser.parse_args()
    compare(args.first_run, args.second_run)
