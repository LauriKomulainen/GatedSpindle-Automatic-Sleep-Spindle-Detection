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
PLOTS_DIR = CURRENT_DIR / "plots"
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

    log.info(f"Loaded stats for {len(stats_dict)} subjects from JSON files.")
    return stats_dict


def plot_spindle_counts(gt_stats, model_stats, save_dir):
    numeric_ids = []
    for item in gt_stats:
        digits = ''.join(filter(str.isdigit, item['id']))
        numeric_ids.append(int(digits) if digits else 999)

    numeric_ids = np.array(numeric_ids)
    sorted_indices = np.argsort(numeric_ids)

    sorted_labels = np.array([str(n) for n in numeric_ids[sorted_indices]])
    sorted_ids = [gt_stats[i]['id'] for i in sorted_indices]

    # Ground Truth data
    s1 = np.array([gt_stats[i]['s1'] for i in sorted_indices])
    s2 = np.array([gt_stats[i]['s2'] for i in sorted_indices])
    union = np.array([gt_stats[i]['union'] for i in sorted_indices])
    kept = np.array([gt_stats[i]['kept'] for i in sorted_indices])

    tp_vals = []
    fp_vals = []

    for sid in sorted_ids:
        m_data = model_stats.get(sid)
        if m_data is None:
            sid_num = ''.join(filter(str.isdigit, sid))
            for k, v in model_stats.items():
                k_num = ''.join(filter(str.isdigit, k))
                if k_num == sid_num and sid_num != "":
                    m_data = v
                    break

        if m_data:
            tp_vals.append(m_data.get('tp', 0))
            fp_vals.append(m_data.get('fn', 0))
        else:
            tp_vals.append(0)
            fp_vals.append(0)

    tp_vals = np.array(tp_vals)
    fp_vals = np.array(fp_vals)

    x = np.arange(len(sorted_labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(16, 8))

    rects1 = ax.bar(x - 1.5 * width, s1, width, label='Expert 1', color='#EFB7B2')

    rects2 = ax.bar(x - 0.5 * width, s2, width, label='Expert 2', color='#6699CC')

    ax.bar(x + 0.5 * width, union, width, label='Total (Raw Union)',
           color='#E0B0FF', edgecolor='#9370DB', linestyle='--', alpha=0.6)
    ax.bar(x + 0.5 * width, kept, width, label='N2/N3 (Filtered)', color='#9370DB')

    rects_tp = ax.bar(x + 1.5 * width, tp_vals, width, label='Model TP (Correct)', color='#20B2AA')
    rects_fp = ax.bar(x + 1.5 * width, fp_vals, width, bottom=tp_vals, label='Model FN (False Neg)', color='#FF7F50')

    ax.set_xlabel('Subject Number')
    ax.set_ylabel('Spindle Count')
    folder_name = MODEL_RUN_DIR.name if MODEL_RUN_DIR else "Unknown"
    ax.set_title(f'Spindle Count Analysis: Experts vs. Ground Truth vs. Model ({folder_name})')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_labels)

    total_model = tp_vals + fp_vals
    max_h = max(np.max(union), np.max(s1), np.max(total_model)) if len(total_model) > 0 else np.max(union)
    ax.set_ylim(0, max_h * 1.35)

    ax.legend(loc='upper center', ncol=6, frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel_inside(rects, text_color='white'):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(rect.get_x() + rect.get_width() / 2, height / 2),
                            ha='center', va='center',
                            fontsize=9, fontweight='bold', color=text_color)

    autolabel_inside(rects1, text_color='white')
    autolabel_inside(rects2, text_color='white')

    for i, u_val in enumerate(union):
        x_pos = x[i] + 0.5 * width
        if u_val > 0:
            ax.annotate(f'{int(u_val)}',
                        xy=(x_pos, u_val),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color='#6A3D9A')

    for i, k_val in enumerate(kept):
        x_pos = x[i] + 0.5 * width
        if k_val > 0:
            y_pos = k_val / 2 if k_val > 10 else k_val + 5
            c = 'white' if k_val > 10 else '#6A3D9A'
            ax.annotate(f'{int(k_val)}',
                        xy=(x_pos, y_pos),
                        ha='center', va='center', fontsize=9, fontweight='bold', color=c)

    for r_tp, r_fp in zip(rects_tp, rects_fp):
        h_tp = r_tp.get_height()
        h_fp = r_fp.get_height()
        total = h_tp + h_fp

        if total > 0:
            ax.annotate(f'{int(total)}',
                        xy=(r_fp.get_x() + r_fp.get_width() / 2, total),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color='#004C4C')
        if h_tp > 8:
            ax.annotate(f'{int(h_tp)}',
                        xy=(r_tp.get_x() + r_tp.get_width() / 2, h_tp / 2),
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        if h_fp > 8:
            ax.annotate(f'{int(h_fp)}',
                        xy=(r_fp.get_x() + r_fp.get_width() / 2, h_tp + h_fp / 2),
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / "spindle_counts_from_json.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    log.info(f"Summary plot saved to {out_file}")


def main():
    if not STATS_FILE_PATH.exists():
        log.error(f"Stats file not found: {STATS_FILE_PATH}")
        return

    log.info(f"Loading Ground Truth stats from {STATS_FILE_PATH}")
    with open(STATS_FILE_PATH, 'r') as f:
        gt_stats = json.load(f)

    model_stats = load_eval_stats_from_folder(MODEL_RUN_DIR)

    plot_spindle_counts(gt_stats, model_stats, PLOTS_DIR)


if __name__ == "__main__":
    main()