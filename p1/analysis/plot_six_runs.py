"""Report figure: train and validation loss of the six runs, one panel per optimizer."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RATE_FIELDS = {'adamw': 'learning_rate', 'muon': 'muon_learning_rate', 'sgdm': 'sgdm_learning_rate'}
TITLES = {'adamw': 'AdamW', 'muon': 'Muon', 'sgdm': 'SGDM'}


def load_run(run_dir):
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'completed':
        raise ValueError(f'Incomplete run: {run_dir.name}')
    with (run_dir / 'metrics.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    method = config['optimizer_name']
    rate = config[RATE_FIELDS[method]] * config['lr_multiplier']
    return dict(name=run_dir.name, method=method, rate=rate, rows=rows, summary=summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', default='runs')
    parser.add_argument('--output', default='runs/six_runs_train_val.png')
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    runs = [load_run(p.parent) for p in sorted(runs_dir.glob('*/summary.json'))]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.7), sharey=True)
    for ax, method in zip(axes, TITLES):
        group = sorted((r for r in runs if r['method'] == method), key=lambda r: r['rate'])
        for run, color in zip(group, ['tab:blue', 'tab:red']):
            xs = [float(row['tokens']) / 1e6 for row in run['rows']]
            ax.plot(xs, [float(row['val_loss']) for row in run['rows']], color=color, marker='o', markersize=3,
                    label=f"val, LR={run['rate']:g}")
            ax.plot(xs, [float(row['train_loss']) for row in run['rows']], color=color, linestyle='--',
                    label=f"train, LR={run['rate']:g}")
        ax.set(title=TITLES[method], xlabel='Training prediction tokens (M)')
        ax.legend(fontsize=7)
    axes[0].set_ylabel('Cross-entropy loss (nats / char)')
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f'Saved: {args.output}')
    print(f"{'run':16s} {'LR':>8s} {'endpoint val':>12s} {'best val':>9s} {'train_s':>8s}")
    for run in sorted(runs, key=lambda r: (list(TITLES).index(r['method']), r['rate'])):
        s = run['summary']
        print(f"{run['name']:16s} {run['rate']:8g} {s['endpoint_scores']['loss']:12.4f} "
              f"{s['best_val_loss']:9.4f} {s['train_seconds']:8.0f}")


if __name__ == '__main__':
    main()
