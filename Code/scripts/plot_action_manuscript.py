"""Plot the saved alpha=10 paired Action replicates for the manuscript."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    ('Simulation gene', 'action_monte_carlo_hermite_primal_vs_wfr_seed1000'),
    ('Mouse hematopoiesis', 'action_monte_carlo_mouse_hermite_primal_vs_wfr_seed1000'),
    ('EMT', 'action_monte_carlo_emt_seed1000'),
]
METHODS = ['RUOT', 'WFR-FM bridge']
COLORS = ['#2675AE', '#DF8841']


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 12, 'axes.labelsize': 11,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8))
    provenance = []
    jitter = np.random.default_rng(42).uniform(-0.13, 0.13, 20)
    for ax, (title, folder) in zip(axes, DATASETS):
        source = ROOT / '../Results' / folder
        summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
        assert summary['alpha'] == 10 and summary['sigma'] == 0.25
        assert summary['mc_seed_range'] == [1000, 1019]
        frame = pd.read_csv(source / 'replicates.csv')
        values = []
        for method in METHODS:
            rows = frame.loc[frame.method == method].sort_values('seed')
            assert rows.seed.tolist() == list(range(1000, 1020))
            assert np.isfinite(rows.total_action).all()
            np.testing.assert_allclose(rows.total_action.mean(), summary['method_summaries'][method]['total_action']['mean'])
            values.append(rows.total_action.to_numpy())
        boxes = ax.boxplot(values, positions=[1, 2], widths=0.48,
                           patch_artist=True, showfliers=False, whis=1.5,
                           medianprops={'color': '#202830', 'linewidth': 1.6},
                           whiskerprops={'color': '#65707D', 'linewidth': 1.1},
                           capprops={'color': '#65707D', 'linewidth': 1.1})
        for i, (patch, color, data) in enumerate(zip(boxes['boxes'], COLORS, values)):
            patch.set(facecolor=color, alpha=0.20, edgecolor=color, linewidth=1.5)
            ax.scatter(i + 1 + jitter, data, color=color, s=22, alpha=0.9,
                       edgecolors='white', linewidths=0.35, zorder=3)
        ax.set_title(title, pad=13, fontweight='semibold')
        ax.set_xticks([1, 2], ['RUOT', 'WFR+BB'])
        ax.set_ylabel('Action')
        ax.set_xlim(0.5, 2.5)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f' if title == 'EMT' else '%.2f'))
        ax.grid(axis='y', color='#E6EAF0', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_color('#AEB6C0')
        ax.tick_params(axis='both', length=3, color='#AEB6C0')
        provenance.append({'dataset': title, 'source': str(source.relative_to(ROOT)),
                           'alpha': 10, 'sigma': .25, 'n_per_method': 20,
                           'seeds': list(range(1000, 1020)),
                           'method_runs': summary['method_runs']})
    fig.subplots_adjust(left=.065, right=.99, top=.85, bottom=.17, wspace=.40)
    destination = ROOT / '../Doc/Experiment1/Figures'
    destination.mkdir(exist_ok=True)
    fig.savefig(destination / 'action_alpha10_three_datasets.pdf', bbox_inches='tight')
    fig.savefig(destination / 'action_alpha10_three_datasets.png', dpi=250, bbox_inches='tight')
    plt.close(fig)
    (destination / 'action_alpha10_sources.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved three-panel boxplots, with all 20 points per method and dataset.')


if __name__ == '__main__':
    main()
