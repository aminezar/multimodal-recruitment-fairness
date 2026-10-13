"""Validate the archived main matrix; emit sample statistics and static PNG.

No training, dataset loading, network access or interactive site. Added during
publication preparation; input is the original thesis's per-run metric CSV.
"""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('none', 'tabular', 'tabular-text', 'tabular-image', 'tabular-image-text')
FUSIONS = ('early', 'mid', 'late')
METRICS = ('mae_biased', 'mae_blind', 'kl', 'parity', 'top100_female', 'equal_opportunity', 'n_selected', 'ndcg')


def summarize(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = {(f, c, str(s)) for f in FUSIONS for c in CONDITIONS for s in range(32, 37)}
    keys = [(r['fusion'], r['condition'], r['seed']) for r in rows]
    if len(rows) != 75 or len(set(keys)) != 75 or set(keys) != expected:
        raise ValueError('Expected exactly 75 unique main runs: three fusions, five conditions, seeds 32-36')
    groups = defaultdict(list)
    for row in rows:
        if row['fusion'] == 'mid' and float(row['text_width']) != 50:
            raise ValueError('Main mid-fusion results must use text width 50')
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError(f'Non-finite {metric}')
            if metric in {'mae_biased', 'mae_blind', 'parity', 'top100_female', 'ndcg'} and not 0 <= value <= 1:
                raise ValueError(f'Out-of-range {metric}')
            if metric == 'equal_opportunity' and not -1 <= value <= 1:
                raise ValueError('Out-of-range equal opportunity gap')
            if metric == 'n_selected' and not 0 <= value <= 4800:
                raise ValueError('Out-of-range selected count')
            groups[row['fusion'], row['condition'], metric].append(value)
    return [dict(fusion=f, condition=c, metric=m, mean=statistics.mean(groups[f, c, m]),
                 std=statistics.stdev(groups[f, c, m]), n_seeds=5)
            for f in FUSIONS for c in CONDITIONS for m in METRICS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'results/results.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'assets')
    args = parser.parse_args()
    summary = summarize(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['fusion', 'condition', 'metric', 'mean', 'std', 'n_seeds'])
        writer.writeheader()
        writer.writerows(summary)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.titlesize': 14,
                         'axes.labelsize': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    lookup = {(r['fusion'], r['condition'], r['metric']): r for r in summary}
    panels = [('kl', 'Score-distribution difference', 'Legacy KL (natural-log units; lower)*'),
              ('parity', 'Top-100 demographic parity', 'Selection-share ratio (1 = parity)'),
              ('equal_opportunity', 'Equal opportunity', 'TPR male − female (0 = parity)'),
              ('ndcg', 'Ranking quality vs blind labels', 'NDCG@10 (higher)'),
              ('mae_biased', 'Fit to biased training targets', 'MAE (score units; lower)'),
              ('mae_blind', 'Accuracy vs blind reference', 'MAE (score units; lower)')]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = ['#2166ac', '#d97706', '#167c52']
    markers = ['o', 's', '^']
    x = np.arange(5)
    for ax, (metric, title, label) in zip(axes.flat, panels):
        for idx, fusion in enumerate(FUSIONS):
            values = [lookup[fusion, c, metric] for c in CONDITIONS]
            ax.errorbar(x + (idx - 1) * .17, [r['mean'] for r in values],
                        yerr=[r['std'] for r in values], fmt=markers[idx], color=colors[idx],
                        markersize=6, capsize=3, elinewidth=1.4, label=f'{fusion.capitalize()} fusion')
        ax.set_title(title, loc='left', weight='bold', pad=12)
        ax.set_ylabel(label)
        ax.set_xticks(x, ['None', 'Tabular', '+ Text', '+ Image', 'All three'])
        ax.set_xlim(-.45, 4.45)
        ax.grid(axis='y', alpha=.22)
        ax.set_axisbelow(True)
        if metric == 'parity':
            ax.axhline(1, color='#666666', lw=.8, ls='--')
            ax.set_ylim(-.08, 1.12)
        elif metric == 'equal_opportunity':
            ax.axhline(0, color='#666666', lw=.8, ls='--')
        elif metric == 'kl':
            ax.set_ylim(bottom=0)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle('Multimodal recruitment fairness · FairCVdb thesis benchmark', x=.055, ha='left', fontsize=20, weight='bold')
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.975, .968), frameon=False, ncol=3)
    fig.subplots_adjust(left=.065, right=.975, top=.85, bottom=.14, wspace=.34, hspace=.45)
    fig.text(.055, .895, 'Archived results · 75 main runs · mid-fusion text width 50 · means ± sample SD across seeds 32–36', fontsize=12, color='#444444')
    fig.text(.055, .065, '“+ Text” and “+ Image” also remove tabular gender; “All three” debiases every modality. Error bars are SD, not confidence intervals.', fontsize=11)
    fig.text(.055, .034, '*Legacy KL omits empty reference bins and scores ≥ 0.98. Synthetic benchmark outcomes are not evidence of fairness in real hiring.', fontsize=11, color='#555555')
    fig.savefig(args.output_dir / 'results_overview.png', dpi=160, facecolor='white', metadata={'Software': 'thesis-publication make_overview.py'})
    plt.close(fig)
    print(f'Validated 75 unique runs; wrote {len(summary)} mean/std rows and results_overview.png')


if __name__ == '__main__':
    main()
