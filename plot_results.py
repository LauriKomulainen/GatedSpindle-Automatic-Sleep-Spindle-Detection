# plot_results.py

import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.logger import setup_logging
import paths

setup_logging("plot_results.log")
log = logging.getLogger(__name__)

LOSO_FOLDER_NAME = "LOSO_run_2025-12-28_12-50-56"
MODEL_RUN_DIR = paths.REPORTS_DIR / LOSO_FOLDER_NAME
CURRENT_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_DIR = paths.PROCESSED_DATA_DIR
PLOTS_DIR = paths.PLOTS_DIR
STATS_FILE_PATH = PROCESSED_DATA_DIR / "subject_stats.json"


def load_eval_stats_from_folder(run_dir: Path):
    stats_dict = {}
    if not run_dir.exists():
        log.error(f"Directory not found: {run_dir}")
        return stats_dict

    log.info(f"Scanning for JSON stats in: {run_dir}")
    json_files = list(run_dir.rglob("eval_stats_*.json"))

    if not json_files:
        log.warning("No 'eval_stats_*.json' files found.")
        return stats_dict

    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            sid = data.get('subject_id')
            if sid:
                stats_dict[sid] = data
        except Exception as e:
            log.error(f"Error reading {jf}: {e}")

    return stats_dict


def prepare_data(gt_stats, model_stats):
    numeric_ids = []
    for item in gt_stats:
        digits = ''.join(filter(str.isdigit, item['id']))
        numeric_ids.append(int(digits) if digits else 999)

    numeric_ids = np.array(numeric_ids)
    sorted_indices = np.argsort(numeric_ids)

    sorted_labels = np.array([str(n) for n in numeric_ids[sorted_indices]])
    sorted_ids = [gt_stats[i]['id'] for i in sorted_indices]

    s1 = np.array([gt_stats[i]['s1'] for i in sorted_indices])
    s2 = np.array([gt_stats[i]['s2'] for i in sorted_indices])
    union = np.array([gt_stats[i]['union'] for i in sorted_indices])
    kept = np.array([gt_stats[i]['kept'] for i in sorted_indices])

    tp_vals = []
    fp_vals = []
    fn_vals = []

    for idx, sid in enumerate(sorted_ids):
        m_data = model_stats.get(sid)

        # Fallback haku
        if m_data is None:
            sid_num = ''.join(filter(str.isdigit, sid))
            for k, v in model_stats.items():
                k_num = ''.join(filter(str.isdigit, k))
                if k_num == sid_num and sid_num != "":
                    m_data = v
                    break

        if m_data:
            tp = m_data.get('tp', 0)
            fp = m_data.get('fp', 0)
            fn = m_data.get('fn', 0)
            true_cnt = m_data.get('true_count', 0)

            tp_vals.append(tp)
            fp_vals.append(fp)
            fn_vals.append(fn)

            if true_cnt > 0:
                kept[idx] = true_cnt
        else:
            tp_vals.append(0)
            fp_vals.append(0)
            fn_vals.append(0)

    return {
        'labels': sorted_labels,
        's1': s1, 's2': s2, 'union': union, 'kept': kept,
        'tp': np.array(tp_vals), 'fp': np.array(fp_vals), 'fn': np.array(fn_vals)
    }


def autolabel_inside(ax, rects, color='white', font_weight='bold'):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            cy = rect.get_y() + height / 2
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, cy),
                        ha='center', va='center',
                        fontsize=9, fontweight=font_weight, color=color)


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


# PLOT 1: ANNOTATION ANALYSIS
def plot_annotation_stats(data, save_dir):
    x = np.arange(len(data['labels']))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 12))

    c_exp1 = '#EFB7B2'
    c_exp2 = '#6699CC'
    c_union = '#E0B0FF'
    c_filt = '#9370DB'

    rects1 = ax.bar(x - 1.5 * width, data['s1'], width, label='Expert 1', color=c_exp1)
    rects2 = ax.bar(x - 0.5 * width, data['s2'], width, label='Expert 2', color=c_exp2)

    ax.bar(x + 0.5 * width, data['union'], width, label='Total (Raw Union)',
           color=c_union, edgecolor=c_filt, linestyle='--', alpha=0.6)

    ax.bar(x + 0.5 * width, data['kept'], width, label='N2/N3 (Filtered)', color=c_filt)

    ax.set_ylabel('Spindle Count')
    ax.set_title('Annotation Analysis (Experts vs. Consensus)')
    ax.set_xticks(x)
    ax.set_xticklabels(data['labels'])

    autolabel_inside(ax, rects1)
    autolabel_inside(ax, rects2)

    for i, (u_val, k_val) in enumerate(zip(data['union'], data['kept'])):
        x_pos = x[i] + 0.5 * width

        if u_val > 0:
            ax.annotate(f'{int(u_val)}', xy=(x_pos, u_val), xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color='#6A3D9A')

        if k_val > 0:
            y_pos = k_val / 2 if k_val > 10 else k_val + 5
            c = 'white' if k_val > 10 else '#6A3D9A'
            ax.annotate(f'{int(k_val)}', xy=(x_pos, y_pos), ha='center', va='center',
                        fontsize=9, fontweight='bold', color=c)

    max_h = max(np.max(data['union']), np.max(data['s1']))

    ax.set_ylim(0, max_h * 1.15)

    ax.legend(loc='upper center', ncol=4)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    out_file = save_dir / "spindle_counts_annotations.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    log.info(f"Saved Annotation Plot to {out_file}")


# PLOT 2: MODEL PERFORMANCE
def plot_model_performance(data, save_dir):
    x = np.arange(len(data['labels']))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 12))

    c_gt = '#9370DB'
    c_tp = '#6699CC'
    c_fn = '#779ECB'
    c_fp = '#EFB7B2'
    c_edge = '#9370DB'

    # 1. Ground Truth
    rects_gt = ax.bar(x - width, data['kept'], width, label='Ground Truth (Target)', color=c_gt)

    # 2. Model TP + FN (Stacked)
    rects_tp = ax.bar(x, data['tp'], width, label='Model Tp', color=c_tp)

    # Model FN
    rects_fn = ax.bar(x, data['fn'], width, bottom=data['tp'], label='Model Fn',
                      color=c_fn, edgecolor=c_edge, linestyle='--', alpha=0.6)

    # 3. Model FP
    rects_fp = ax.bar(x + width, data['fp'], width, label='Model Fp', color=c_fp)

    ax.set_ylabel('Spindle Count')
    ax.set_title('Model Performance (Ground Truth vs. Predictions breakdown)')
    ax.set_xticks(x)
    ax.set_xticklabels(data['labels'])
    ax.set_xlabel('Subject Number')

    autolabel_smart_stacked(ax, rects_gt, text_color_small='white')
    autolabel_smart_stacked(ax, rects_tp, text_color_small='white')
    autolabel_smart_stacked(ax, rects_fn, text_color_small='#003366')
    autolabel_smart_stacked(ax, rects_fp, text_color_small='white')

    max_h = max(np.max(data['kept']), np.max(data['fp']))

    ax.set_ylim(0, max_h * 1.15)

    ax.legend(loc='upper center', ncol=4)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    out_file = save_dir / "spindle_counts_model_performance.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    log.info(f"Saved Model Performance Plot to {out_file}")


def main():
    if not STATS_FILE_PATH.exists():
        log.error("Stats file missing.")
        return

    with open(STATS_FILE_PATH, 'r') as f:
        gt_stats = json.load(f)

    model_stats = load_eval_stats_from_folder(MODEL_RUN_DIR)

    data = prepare_data(gt_stats, model_stats)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_annotation_stats(data, PLOTS_DIR)
    plot_model_performance(data, PLOTS_DIR)

if __name__ == "__main__":
    main()