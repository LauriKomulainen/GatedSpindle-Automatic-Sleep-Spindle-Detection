# plot_results.py

import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from utils.logger import setup_logging
import paths
import matplotlib.ticker as ticker

setup_logging("plots.log")
log = logging.getLogger(__name__)

# CONFIGURATION
LOSO_FOLDER_NAME = "Hilbert_best"
TARGET_REPEAT = "Repeat_1"

# PATHS
MODEL_RUN_DIR = paths.REPORTS_DIR / LOSO_FOLDER_NAME
STATS_FILE_PATH = paths.PROCESSED_DATA_DIR / "subject_stats.json"
PLOTS_DIR = paths.PLOTS_DIR


def clean_subject_id(sid):
    """Removes model suffixes to get pure subject ID (e.g. 'excerpt1_best' -> 'excerpt1')."""
    return sid.replace('_ens', '').replace('_best', '').replace('_swa', '')


def load_all_experiment_data(experiment_dir: Path):
    """
    Scans the entire experiment folder structure and loads evaluation stats.
    Structure: Repeat_X -> Fold_Y -> eval_stats_SUBJECT_MODE.json

    Returns:
        data[subject_id][repeat_name] = stats_dict
    """
    if not experiment_dir.exists():
        log.error(f"Experiment directory not found: {experiment_dir}")
        return {}

    log.info(f"Scanning experiment data from: {experiment_dir}")
    data = defaultdict(dict)

    # Iterate Repeat directories
    repeat_dirs = sorted(list(experiment_dir.glob("Repeat_*")))
    if not repeat_dirs:
        log.warning("No Repeat directories found. Checking root...")
        repeat_dirs = [experiment_dir]

    for rep_dir in repeat_dirs:
        repeat_name = rep_dir.name

        # Iterate Fold directories inside Repeat
        fold_dirs = sorted(list(rep_dir.glob("Fold_*")))

        for fold_dir in fold_dirs:
            # Find eval_stats json. Prioritize Ensemble > SWA > Best if multiple exist.
            json_files = list(fold_dir.glob("eval_stats_*.json"))
            if not json_files:
                continue

            selected_file = None
            # Priority check
            for mode in ['_ens', '_swa', '_best']:
                candidates = [f for f in json_files if f.stem.endswith(mode)]
                if candidates:
                    selected_file = candidates[0]
                    break

            # Fallback
            if not selected_file:
                selected_file = json_files[0]

            try:
                with open(selected_file, 'r') as f:
                    stats = json.load(f)

                raw_sid = stats.get('subject_id', 'unknown')
                sid = clean_subject_id(raw_sid)
                data[sid][repeat_name] = stats

            except Exception as e:
                log.error(f"Error loading {selected_file}: {e}")

    log.info(f"Loaded data for {len(data)} subjects across {len(repeat_dirs)} repeats.")
    return data


def prepare_metrics_data_aggregated(experiment_data):
    """Calculates Mean & Std of metrics across available repeats for each subject."""
    plot_data = {
        'labels': [],
        'f1_mean': [], 'f1_std': [],
        'prec_mean': [], 'prec_std': [],
        'rec_mean': [], 'rec_std': [],
        'miou_mean': [], 'miou_std': []
    }

    # Collectors for Grand Average
    all_f1, all_prec, all_rec, all_miou = [], [], [], []

    for sid in sorted(experiment_data.keys()):
        repeats = experiment_data[sid].values()
        if not repeats: continue

        # Extract values
        f1s = [r.get('f1', 0) for r in repeats]
        precs = [r.get('precision', 0) for r in repeats]
        recs = [r.get('recall', 0) for r in repeats]
        mious = [r.get('mean_iou', 0) for r in repeats]

        # Calc Stats
        plot_data['labels'].append(sid)
        plot_data['f1_mean'].append(np.mean(f1s))
        plot_data['f1_std'].append(np.std(f1s))
        plot_data['prec_mean'].append(np.mean(precs))
        plot_data['prec_std'].append(np.std(precs))
        plot_data['rec_mean'].append(np.mean(recs))
        plot_data['rec_std'].append(np.std(recs))
        plot_data['miou_mean'].append(np.mean(mious))
        plot_data['miou_std'].append(np.std(mious))

        all_f1.append(np.mean(f1s))
        all_prec.append(np.mean(precs))
        all_rec.append(np.mean(recs))
        all_miou.append(np.mean(mious))

    return plot_data


def prepare_performance_counts(experiment_data, target_repeat, gt_stats):
    """Prepares TP/FP/FN counts for a specific repeat vs Ground Truth."""
    # Helper to sort by numeric ID
    numeric_ids = []
    for item in gt_stats:
        digits = ''.join(filter(str.isdigit, item['id']))
        numeric_ids.append(int(digits) if digits else 999)

    sorted_indices = np.argsort(numeric_ids)

    # Prepare Arrays
    data = {
        'labels': np.array([str(numeric_ids[i]) for i in sorted_indices]),
        's1': np.array([gt_stats[i]['s1'] for i in sorted_indices]),
        's2': np.array([gt_stats[i]['s2'] for i in sorted_indices]),
        'union': np.array([gt_stats[i]['union'] for i in sorted_indices]),
        'kept': np.array([gt_stats[i]['kept'] for i in sorted_indices]),
        'tp': [], 'fp': [], 'fn': []
    }

    sorted_ids = [gt_stats[i]['id'] for i in sorted_indices]

    for idx, sid in enumerate(sorted_ids):
        # clean sid for lookup
        lookup_sid = clean_subject_id(sid)

        # Check if we have data for this subject and repeat
        subj_data = experiment_data.get(lookup_sid, {})
        rep_data = subj_data.get(target_repeat)

        if rep_data:
            data['tp'].append(rep_data.get('tp', 0))
            data['fp'].append(rep_data.get('fp', 0))
            data['fn'].append(rep_data.get('fn', 0))

            # Update ground truth count from model file just in case filtering differed
            true_cnt = rep_data.get('true_count', 0)
            if true_cnt > 0:
                data['kept'][idx] = true_cnt
        else:
            data['tp'].append(0)
            data['fp'].append(0)
            data['fn'].append(0)

    data['tp'] = np.array(data['tp'])
    data['fp'] = np.array(data['fp'])
    data['fn'] = np.array(data['fn'])

    return data


def autolabel_inside(ax, rects, color='black', fontsize=12):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            cy = rect.get_y() + height / 2
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, cy),
                        ha='center', va='center',
                        fontsize=fontsize, color=color)


def autolabel_smart_stacked(ax, rects, text_color_small='#000000'):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            cy = rect.get_y() + height / 2
            c = text_color_small
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, cy),
                        ha='center', va='center',
                        fontsize=9, fontweight='bold', color=c)


def plot_annotation_stats(data, save_dir):
    x = np.arange(len(data['labels']))
    width = 0.25
    fig, ax = plt.subplots(figsize=(16, 12))

    c_exp1, c_exp2 = '#EFB7B2', '#6699CC'
    c_exp2 = '#6699CC'
    c_filt = '#9370DB'

    rects1 = ax.bar(x - 1.5 * width, data['s1'], width, label='Expert 1', color=c_exp1)
    rects2 = ax.bar(x - 0.5 * width, data['s2'], width, label='Expert 2', color=c_exp2)

    rects_union = ax.bar(x + 0.5 * width, data['union'], width, label='Union',
                         color=c_filt, edgecolor=c_filt, linestyle='-', linewidth=2, alpha=0.5)

    ax.set_ylabel('Sleep spindle count', fontsize=16)
    ax.set_xlabel('Excerpts', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(data['labels'], fontsize=16)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.tick_params(axis='y', labelsize=14)

    autolabel_inside(ax, rects1, fontsize=12)
    autolabel_inside(ax, rects2, fontsize=12)

    for rect in rects_union:
        height = rect.get_height()
        if height > 0:
            y_pos = rect.get_y() + height / 2

            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                        ha='center', va='center',
                        fontsize=12)  # Musta teksti

    max_h = max(np.max(data['union']), np.max(data['s1'])) if len(data['union']) > 0 else 10
    ax.set_ylim(0, max_h * 1.15)

    ax.legend(loc='upper center', ncol=4, fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_file = save_dir / "spindle_counts_annotations.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.05)

    plt.close()
    log.info(f"Saved Annotation Plot to {out_file}")

def plot_filtered_union(data, save_dir):
    x = np.arange(len(data['labels']))
    width = 0.25
    fig, ax = plt.subplots(figsize=(16, 12))

    filtered_union_n2_n3 = '#F4A460'
    union_of_two_experts = '#9370DB'

    rects_union = ax.bar(x - width / 2, data['union'], width,
                         label='Union (All sleep stages)',
                         color=union_of_two_experts, edgecolor=union_of_two_experts, linestyle='-', alpha=0.7)

    # Filtered Union (N2/N3)
    rects_filt = ax.bar(x + width / 2, data['kept'], width,
                        label='Union (N2/N3 Filtered)',
                        color=filtered_union_n2_n3, edgecolor=filtered_union_n2_n3, linestyle='-', alpha=0.5)


    ax.set_ylabel('Sleep spindle count', fontsize=16)
    ax.set_xlabel('Excerpts', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(data['labels'], fontsize=16)

    ax.set_xticks(x)
    ax.set_xticklabels(data['labels'], fontsize=14)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    def label_bars(rects, text_color):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                y_pos = height / 2
                ax.annotate(f'{int(height)}',
                            xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                            ha='center', va='center',
                            fontsize=14, color=text_color)

    label_bars(rects_union, text_color='black')
    label_bars(rects_filt, text_color='black')

    ax.set_ylabel('Sleep spindle count', fontsize=16)
    ax.set_xlabel('Excerpts', fontsize=16)

    ax.set_xticks(x)
    ax.set_xticklabels(data['labels'], fontsize=14)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

    max_h = np.max(data['union']) if len(data['union']) > 0 else 10
    ax.set_ylim(0, max_h * 1.15)

    ax.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True, shadow=True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_file = save_dir / "filtered_spindle_counts.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.05)

    plt.close()
    log.info(f"Saved Filtered Union Plot to {out_file}")

def plot_model_performance(data, save_dir, folder_name="", repeat_name=""):
    x = np.arange(len(data['labels']))
    width = 0.25
    fig, ax = plt.subplots(figsize=(16, 12))

    c_gt, c_tp, c_fn, c_fp = '#9370DB', '#6699CC', '#779ECB', '#EFB7B2'
    c_edge = '#9370DB'

    rects_gt = ax.bar(x - width, data['kept'], width, label='Ground Truth (Target)', color=c_gt)
    rects_tp = ax.bar(x, data['tp'], width, label='Model Tp', color=c_tp)
    rects_fn = ax.bar(x, data['fn'], width, bottom=data['tp'], label='Model Fn', color=c_fn, edgecolor=c_edge,
                      linestyle='--', alpha=0.6)
    rects_fp = ax.bar(x + width, data['fp'], width, label='Model Fp', color=c_fp)

    ax.set_ylabel('Spindle Count')
    ax.set_title(f'Model Performance (Ground Truth & Model Predictions)\nSource: {folder_name} | {repeat_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(data['labels'])
    ax.set_xlabel('Subject Number')

    autolabel_smart_stacked(ax, rects_gt, text_color_small='white')
    autolabel_smart_stacked(ax, rects_tp, text_color_small='white')
    autolabel_smart_stacked(ax, rects_fn, text_color_small='#003366')
    autolabel_smart_stacked(ax, rects_fp, text_color_small='white')

    max_val = max(np.max(data['kept']), np.max(data['fp'])) if len(data['kept']) > 0 else 10
    ax.set_ylim(0, max_val * 1.15)
    ax.legend(loc='upper center', ncol=4)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    out_file = save_dir / "spindle_counts_model_performance.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    log.info(f"Saved Model Performance Plot ({repeat_name}) to {out_file}")


def plot_metric_statistics(stats_data, save_dir):
    labels = stats_data['labels']
    # Clean labels for display
    clean_labels = []
    for l in labels:
        if "Average" in l:
            clean_labels.append("Average")
        else:
            txt = l.replace('excerpt', 'Excerpt ')
            clean_labels.append(txt)

    means = {
        'F1-Score': stats_data['f1_mean'], 'Precision': stats_data['prec_mean'],
        'Recall': stats_data['rec_mean'], 'mIoU': stats_data['miou_mean']
    }
    stds = {
        'F1-Score': stats_data['f1_std'], 'Precision': stats_data['prec_std'],
        'Recall': stats_data['rec_std'], 'mIoU': stats_data['miou_std']
    }

    x = np.arange(len(clean_labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(18, 10))

    metrics_info = [
        ('F1-Score', x - 1.5 * width, '#EFB7B2'),
        ('Precision', x - 0.5 * width, '#6699CC'),
        ('Recall', x + 0.5 * width, '#E0B0FF'),
        ('mIoU', x + 1.5 * width, '#9370DB')
    ]

    for name, pos, color in metrics_info:
        m_vals = means[name]
        s_vals = stds[name]
        for i, label in enumerate(labels):
            ax.bar(pos[i], m_vals[i], width, label=name if i == 0 else "", color=color, alpha=0.9,
                   yerr=s_vals[i], capsize=5, ecolor='black')

    ax.set_title('Model performance metrics per subject (average across repeats)')
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, fontsize=11, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    if 'Average' in labels:
        idx = labels.index('Average')
        ax.axvspan(idx - 0.5, idx + 0.5, color='gray', alpha=0.1, zorder=0)

    out_file = save_dir / "performance_metrics.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved Metrics Plot to {out_file}")


def main():
    # 1. Load Ground Truth Stats
    if not STATS_FILE_PATH.exists():
        log.error(f"Stats file missing: {STATS_FILE_PATH}")
        return

    with open(STATS_FILE_PATH, 'r') as f:
        gt_stats = json.load(f)

    # 2. Load Experiment Data
    experiment_data = load_all_experiment_data(MODEL_RUN_DIR)
    if not experiment_data:
        log.error("No experiment data loaded. Exiting.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Plot 1: Annotations (Pure GT)
    data_gt_only = prepare_performance_counts({}, "", gt_stats)  # Dummy exp data to get GT structure
    plot_annotation_stats(data_gt_only, PLOTS_DIR)
    plot_filtered_union(data_gt_only, PLOTS_DIR)

    # 4. Plot 2: Model Performance (Counts for specific Repeat)
    data_counts = prepare_performance_counts(experiment_data, TARGET_REPEAT, gt_stats)
    plot_model_performance(data_counts, PLOTS_DIR, folder_name=LOSO_FOLDER_NAME, repeat_name=TARGET_REPEAT)

    # 5. Plot 3: Metrics (Aggregated Mean/Std)
    data_metrics = prepare_metrics_data_aggregated(experiment_data)
    plot_metric_statistics(data_metrics, PLOTS_DIR)

    log.info("All plots generated successfully.")


if __name__ == "__main__":
    main()