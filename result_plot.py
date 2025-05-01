import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# common ordering
VARIANT_ORDER = ['original','pure','LoRA_whole','LoRA_AR','LoRA_NAR']

import matplotlib.colors as mcolors

def darken_color(color, factor=0.6):
    rgb = mcolors.to_rgb(color)
    return tuple(factor * c for c in rgb)

def plot_grouped_bars_multi(df, label_col, metrics, metric_labels, title, ylabel, out_name):
    # reorder so ‘original’ is first
    df = df.set_index(label_col).loc[VARIANT_ORDER].reset_index()

    n_labels    = len(df)
    x           = np.arange(n_labels)
    n_metrics   = len(metrics)
    width       = 0.8 / n_metrics
    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (col, lbl) in enumerate(zip(metrics, metric_labels)):
        offsets = x - 0.4 + width/2 + i*width
        base = base_colors[i % len(base_colors)]
        for j, val in enumerate(df[col]):
            is_orig = (df[label_col][j] == 'original')
            color   = darken_color(base) if is_orig else base
            # only label the first experimental bar (j==1)
            legend_label = lbl if j == 1 else ""
            ax.bar(offsets[j], val, width, color=color, label=legend_label)
            ax.text(
                offsets[j], val, f"{val:.2f}",
                ha='center', va='bottom', fontsize=9
            )

    # divider after original cluster
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(df[label_col], rotation=45, ha='right')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    plt.savefig(out_name, dpi=150)
    plt.close()



def plot_single_bar(df, label_col, val_col, title, ylabel, out_name):
    # reorder so 'original' is first
    df = df.set_index(label_col).loc[VARIANT_ORDER].reset_index()

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, val in enumerate(df[val_col]):
        col = 'grey' if df[label_col][i] == 'original' else 'blue'
        bar = ax.bar(x[i], val, width=0.6, color=col, alpha=0.7)
        ax.text(
            x[i], val,
            f"{val:.2f}",
            ha='center', va='bottom', fontsize=9
        )

    # divider between original and the rest
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(df[label_col], rotation=45, ha='right')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.savefig(out_name, dpi=150)
    plt.close()


def plot_for_file(csv_path, lang):
    df = pd.read_csv(csv_path)
    grouped = df.groupby(['lang','variant'], as_index=False).mean(numeric_only=True)
    grouped['label'] = grouped['variant']

    # relative errors
    grouped['rel_err_f0']     = np.abs(grouped['f0_ref_mean'] - grouped['f0_syn_mean']) / grouped['f0_ref_mean'] * 100
    grouped['rel_err_F1']     = np.abs(grouped['F1_ref']       - grouped['F1_syn'])       / grouped['F1_ref'] * 100
    grouped['rel_err_jitter'] = np.abs(grouped['jitter_ref']   - grouped['jitter_syn'])   / grouped['jitter_ref'] * 100

    # single‐bars
    plot_single_bar(grouped, 'label', 'rel_err_f0',
                    f'{lang.upper()}: Relative Error in F0 Mean',
                    '|ref−syn|/ref (%)', f'{lang}_rel_err_f0.png')

    plot_single_bar(grouped, 'label', 'rel_err_F1',
                    f'{lang.upper()}: Relative Error in F1 Mean',
                    '|ref−syn|/ref (%)', f'{lang}_rel_err_f1.png')

    plot_single_bar(grouped, 'label', 'rel_err_jitter',
                    f'{lang.upper()}: Relative Error in Jitter',
                    '|ref−syn|/ref (%)', f'{lang}_rel_err_jitter.png')

    # # multi‐bars: grammar fidelity
    # plot_grouped_bars_multi(
    #     grouped, 'label',
    #     ['prec_ASR','rec_ASR','f1_ASR'],
    #     ['Precision','Recall','F1 Score'],
    #     f'{lang.upper()}: ASR Grammar‐Fidelity by Variant',
    #     'Score', f'{lang}_grammar_fidelity.png'
    # )

    # PESQ
    if 'pesq' in grouped:
        plot_single_bar(grouped, 'label', 'pesq',
                        f'{lang.upper()}: PESQ Scores by Variant',
                        'PESQ', f'{lang}_pesq.png')

def main():
    # plot_for_file('results/summary_en.csv', lang='en')
    plot_for_file('results/summary_jp.csv', lang='jp')
if __name__ == '__main__':
    main()
