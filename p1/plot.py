"""Read metrics.csv and plot both losses against actual tokens and training time."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_records import load_runs
from exercise_helpers import todo


def draw_loss_curves(ax, rows, x_key, x_label, series, **line_options):
    """series contains (CSV column, legend label) pairs. Same TODO for all figures."""
    for field, label in series:
        xs = [float(row[x_key]) for row in rows]
        ys = [float(row[field]) for row in rows]
        # TODO plots: Plot each (CSV column, label) in series against x_key on ax, forwarding line_options; label axes and add a legend.
        # Fill only the x/y arguments with the prepared xs/ys, respectively.
        # Keep label, marker, markersize and **line_options. Axis labels, legend,
        # CSV aggregation, evaluation and file saving are provided outside this block.
        ax.plot(xs, ys,
                label=label, marker='o', markersize=3, **line_options)
    ax.set(xlabel=x_label, ylabel='Cross-entropy loss (nats / character)')
    ax.legend(fontsize=8)


def plot_run(run_dir):
    run_dir = Path(run_dir)
    with (run_dir / 'metrics.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('No recorded evaluation rows')
    for x_key, x_label in (('tokens', 'Training prediction tokens'), ('train_seconds', 'Training time (s)')):
        fig, ax = plt.subplots()
        draw_loss_curves(ax, rows, x_key, x_label, [('train_loss', 'train'), ('val_loss', 'validation')])
        fig.tight_layout()
        fig.savefig(run_dir / f'loss_vs_{x_key}.png', dpi=150)
        plt.close(fig)


def plot_sweep(sweep_dir):
    """Compare recorded trajectories only; no retraining or automatic crossover claim."""
    sweep_dir = Path(sweep_dir)
    trials = load_runs(sweep_dir)
    groups, warnings, images = {}, [], []
    for trial in trials:
        # Prefer colocated run folders so a downloaded result archive stays usable.
        run_dir = sweep_dir / Path(trial['run_dir'].replace('\\', '/')).name
        config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
        summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
        if summary['status'] != 'completed':
            raise ValueError(f'Incomplete run: {run_dir.name}')
        with (run_dir / 'metrics.csv').open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f'No measurements: {run_dir.name}')
        if len(rows) < 3:
            warnings.append(f'{run_dir.name}: fewer than 3 recorded points; cannot assess an intermediate crossover.')
        method, multiplier = config['optimizer_name'], config['lr_multiplier']
        base = config[{'adamw':'learning_rate', 'muon':'muon_learning_rate', 'sgdm':'sgdm_learning_rate'}[method]]
        rate_label='peak LR' if config.get('lr_schedule')=='cosine' else 'LR'
        rate = base * multiplier
        label = f'{method}, {rate_label}={rate:g}'
        groups.setdefault(trial['protocol'], []).append((method, rate, label, rows))
    colors = {'adamw':'tab:blue', 'muon':'tab:orange', 'sgdm':'tab:green'}
    for protocol, runs in groups.items():
        for x_key, x_label in (('tokens', 'Training prediction tokens'), ('train_seconds', 'Training time (s)')):
            fig, ax = plt.subplots(figsize=(8, 5))
            for method, rate, label, rows in runs:
                rates = sorted({r[1] for r in runs if r[0] == method})
                style = ['-', '--', ':', '-.'][rates.index(rate) % 4]
                draw_loss_curves(ax, rows, x_key, x_label, [('val_loss', label)],
                                 color=colors[method], linestyle=style)
            ax.set_title(f'Validation trajectories: fixed-{protocol} runs')
            fig.tight_layout()
            filename = f'comparison_{protocol}_vs_{x_key}.png'
            fig.savefig(sweep_dir / filename, dpi=150)
            plt.close(fig)
            images.append(filename)
    report = {'images':images, 'warnings':warnings,
              'note':'Lines connect measured points. Inspect ordering changes within the recorded range only; no crossing is required.'}
    (sweep_dir / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run-dir')
    mode.add_argument('--runs-dir', '--sweep-dir', dest='runs_dir')
    args = parser.parse_args()
    if args.runs_dir:
        print(json.dumps(plot_sweep(args.runs_dir), indent=2))
    else:
        plot_run(args.run_dir)
