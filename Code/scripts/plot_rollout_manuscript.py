"""Replay trained RUOT rollouts and plot genuine continuous SDE trajectories."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.runtime import configure, load_config, snapshots_for, file_hash
from src.matching import make_feedback, validate_feedback_architecture
from src.sample import rollout

RUNS = [('Simulation gene', 'gene_distance_hermite_primal_seed42'),
        ('Mouse hematopoiesis', 'mouse_distance_hermite_primal_seed42'),
        ('EMT', 'emt_distance_hermite_primal_seed42')]


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9))
    provenance = []
    destination = ROOT / '../Doc/Experiment1/Figures'
    destination.mkdir(exist_ok=True)
    for ax, (title, folder) in zip(axes, RUNS):
        run = ROOT / '../Results' / folder
        config = load_config(run / 'config.yaml')
        assert config['growth_penalty'] == 10 and config['model_sigma'] == .25
        device = configure(config)
        truth = snapshots_for(config, full=True)
        model = make_feedback(config, device)
        checkpoint = torch.load(run / 'feedback.pt', map_location=device, weights_only=False)
        validate_feedback_architecture(checkpoint, config)
        for key, file in [('path_checkpoint_sha256', 'pair_path.pt'),
                          ('coupling_manifest_sha256', 'couplings/manifest.json')]:
            assert checkpoint[key] == file_hash(run / file)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        gen = config['generation']
        display_rng = np.random.default_rng(73)
        trace_ids = display_rng.choice(gen['n_particles'], 8, replace=False)
        result = rollout(model, truth[0].x, config['model_times'], config['model_sigma'],
                         gen['n_particles'], gen['steps_per_unit_interval'], config['seed']+300,
                         gen['endpoint_guard'], trace_indices=trace_ids)
        with np.load(run / 'generated.npz') as saved:
            error = float(np.max(np.abs(result['positions']-saved['positions'])))
            weight_error = float(np.max(np.abs(result['log_weights']-saved['log_weights'])))
        assert error == 0 and weight_error == 0, (folder, error, weight_error)
        np.savez_compressed(destination / f'{folder}_traces.npz',
                            trace_positions=result['trace_positions'], trace_times=result['trace_times'],
                            trace_indices=trace_ids, feedback_checkpoint_sha256=file_hash(run/'feedback.pt'))
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(truth)))
        time_handles = []
        for k, (snapshot, color) in enumerate(zip(truth, colors)):
            ax.scatter(snapshot.x[:, 0], snapshot.x[:, 1], marker='x', color=color,
                       s=12, linewidths=.65, alpha=.20, rasterized=True, zorder=1)
            ids = display_rng.choice(gen['n_particles'], 80, replace=False)
            positions = result['positions'][k, ids]
            weights = np.exp(result['log_weights'][k]-np.max(result['log_weights'][k]))
            ax.scatter(positions[:, 0], positions[:, 1], marker='o', color=color,
                       edgecolors='#202020', linewidths=.55, s=18*weights[ids]/weights.mean(),
                       alpha=.75, rasterized=True, zorder=2)
            time_handles.append(Line2D([], [], marker='o', color=color, linewidth=.8,
                                       markersize=4, label=f'T{snapshot.time:g}'))
        for line in result['trace_positions'].transpose(1, 0, 2):
            ax.plot(line[:, 0], line[:, 1], color='#F01818', lw=.55, alpha=.8, zorder=3)
        ax.set_title(title, pad=10, fontweight='semibold')
        ax.set_xlabel('$x_1$' if config['dimension']==2 else '$x_1$ (10D projection)')
        ax.set_ylabel('$x_2$')
        ax.legend(handles=time_handles, loc='upper left', fontsize=7, framealpha=.88,
                  handlelength=1.2, handletextpad=.4, borderpad=.3, labelspacing=.25)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#C0C5CC')
        ax.tick_params(length=3, color='#AEB6C0', labelsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.margins(.07)
        provenance.append(dict(dataset=title, run=folder, seed=config['seed']+300,
                               dimension=config['dimension'], trace_ids=trace_ids.tolist(),
                               displayed_predicted_per_time=80, snapshot_replay_max_error=error,
                               log_weight_replay_max_error=weight_error,
                               feedback_checkpoint_sha256=file_hash(run/'feedback.pt')))
    fig.legend(handles=[Line2D([],[],marker='x',linestyle='',color='#727B87',label='Ground truth'),
                        Line2D([],[],marker='o',linestyle='',color='#71BC8A',markeredgecolor='#202020',label='Predicted'),
                        Line2D([],[],color='#F01818',linewidth=1,label='Trajectory')],
               loc='lower center', ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(.5,-.015))
    fig.subplots_adjust(left=.065,right=.99,top=.87,bottom=.22,wspace=.36)
    fig.savefig(destination/'ruot_rollouts_three_datasets.pdf',bbox_inches='tight',dpi=300)
    fig.savefig(destination/'ruot_rollouts_three_datasets.png',bbox_inches='tight',dpi=250)
    plt.close(fig)
    (destination/'ruot_rollout_sources.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(provenance,indent=2))


if __name__ == '__main__':
    main()
