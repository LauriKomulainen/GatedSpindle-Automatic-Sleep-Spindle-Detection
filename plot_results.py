# plot_results.py

import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.logger import setup_logging
import paths

setup_logging("plots.log")
log = logging.getLogger(__name__)

# --- Configuration ---
LOSO_FOLDER_NAME = "hilbert_best"
TARGET_REPEAT = "Repeat_2"
# ---------------------

MODEL_RUN_DIR = paths.REPORTS_DIR / LOSO_FOLDER_NAME
CURRENT_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_DIR = paths.PROCESSED_DATA_DIR
PLOTS_DIR = paths.PLOTS_DIR
STATS_FILE_PATH = PROCESSED_DATA_DIR / "subject_stats.json"


def clean_subject_id(sid):
    return sid.replace('_ens', '').replace('_best', '').replace('_swa', '')


def load_single_repeat_stats(experiment_root_dir: Path, repeat_name: str):
    target_dir = experiment_root_dir / repeat_name

    if not target_dir.exists():
        log.error(f"Target repeat directory not found: {target_dir}")
        return {}

    log.info(f"Loading single repeat stats from: {target_dir}")
    stats = {}

    json_files = list(target_dir.rglob("eval_stats_*_ens.json"))
    if not json_files:
        json_files = list(target_dir.rglob("eval_stats_*_best.json"))

    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            sid = data.get('subject_id')
            if sid:
                clean_sid = clean_subject_id(sid)
                stats[clean_sid] = data
        except Exception as e:
            log.error(f"Error reading {jf}: {e}")

    return stats


def load_eval_stats_from_folder(experiment_root_dir: Path):
    if not experiment_root_dir.exists():
        log.error(f"Directory not found: {experiment_root_dir}")
        return {}

    log.info(f"Scanning for Repeats (Aggregate) in: {experiment_root_dir}")

    aggregated_data = {}
    repeat_dirs = sorted(list(experiment_root_dir.glob("Repeat_*")))

    if not repeat_dirs:
        repeat_dirs = [experiment_root_dir]

    for rep_dir in repeat_dirs:
        json_files = list(rep_dir.rglob("eval_stats_*_ens.json"))
        if not json_files:
            json_files = list(rep_dir.rglob("eval_stats_*_best.json"))

        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                sid = data.get('subject_id')
                if sid:
                    clean_sid = clean_subject_id(sid)

                    if clean_sid not in aggregated_data:
                        aggregated_data[clean_sid] = []
                    aggregated_data[clean_sid].append(data)
            except Exception as e:
                log.error(f"Error reading {jf}: {e}")

    plot_data = {
        'labels': [],
        'f1_mean': [], 'f1_std': [],
        'prec_mean': [], 'prec_std': [],
        'rec_mean': [], 'rec_std': [],
        'miou_mean': [], 'miou_std': []
    }

    all_subjects_f1 = []
    all_subjects_prec = []
    all_subjects_rec = []
    all_subjects_miou = []

    averaged_stats = {}

    for sid, run_list in sorted(aggregated_data.items()):
        if not run_list:
            continue

        base_stats = run_list[0].copy()

        f1s = [r.get('f1', 0) for r in run_list]
        precs = [r.get('precision', 0) for r in run_list]
        recs = [r.get('recall', 0) for r in run_list]
        mious = [r.get('mean_iou', 0) for r in run_list]

        m_f1, s_f1 = np.mean(f1s), np.std(f1s)
        m_prec, s_prec = np.mean(precs), np.std(precs)
        m_rec, s_rec = np.mean(recs), np.std(recs)
        m_miou, s_miou = np.mean(mious), np.std(mious)

        plot_data['labels'].append(sid)
        plot_data['f1_mean'].append(m_f1)
        plot_data['f1_std'].append(s_f1)
        plot_data['prec_mean'].append(m_prec)
        plot_data['prec_std'].append(s_prec)
        plot_data['rec_mean'].append(m_rec)
        plot_data['rec_std'].append(s_rec)
        plot_data['miou_mean'].append(m_miou)
        plot_data['miou_std'].append(s_miou)

        all_subjects_f1.append(m_f1)
        all_subjects_prec.append(m_prec)
        all_subjects_rec.append(m_rec)
        all_subjects_miou.append(m_miou)

        averaged_stats[sid] = base_stats

    if all_subjects_f1:
        avg_label = "Average"
        gm_f1, gs_f1 = np.mean(all_subjects_f1), np.std(all_subjects_f1)
        gm_prec, gs_prec = np.mean(all_subjects_prec), np.std(all_subjects_prec)
        gm_rec, gs_rec = np.mean(all_subjects_rec), np.std(all_subjects_rec)
        gm_miou, gs_miou = np.mean(all_subjects_miou), np.std(all_subjects_miou)

        plot_data['labels'].append(avg_label)
        plot_data['f1_mean'].append(gm_f1)
        plot_data['f1_std'].append(gs_f1)
        plot_data['prec_mean'].append(gm_prec)
        plot_data['prec_std'].append(gs_prec)
        plot_data['rec_mean'].append(gm_rec)
        plot_data['rec_std'].append(gs_rec)
        plot_data['miou_mean'].append(gm_miou)
        plot_data['miou_std'].append(gs_miou)

    plot_save_dir = paths.PLOTS_DIR
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_statistics(plot_data, plot_save_dir)

    log.info(f"Aggregated stats for {len(averaged_stats)} subjects.")
    return averaged_stats


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
    ax.set_title('Annotation Analysis (Experts & Consensus)')
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
def plot_model_performance(data, save_dir, folder_name="", repeat_name=""):
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

    # Otsikko dynaamisesti
    title_text = 'Model Performance (Ground Truth & Model Predictions)'
    if folder_name and repeat_name:
        title_text += f"\nSource: {folder_name} | {repeat_name}"

    ax.set_title(title_text)
    ax.set_xticks(x)
    ax.set_xticklabels(data['labels'])
    ax.set_xlabel('Subject Number')

    autolabel_smart_stacked(ax, rects_gt, text_color_small='white')
    autolabel_smart_stacked(ax, rects_tp, text_color_small='white')
    autolabel_smart_stacked(ax, rects_fn, text_color_small='#003366')
    autolabel_smart_stacked(ax, rects_fp, text_color_small='white')

    # Skaalataan Y-akseli
    max_val = 0
    if len(data['kept']) > 0:
        max_val = max(max_val, np.max(data['kept']))
    if len(data['fp']) > 0:
        max_val = max(max_val, np.max(data['fp']))

    ax.set_ylim(0, max_val * 1.15)

    ax.legend(loc='upper center', ncol=4)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    out_file = save_dir / "spindle_counts_model_performance.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    log.info(f"Saved Model Performance Plot ({repeat_name}) to {out_file}")


# PLOT 3: METRIC STATISTICS
def plot_metric_statistics(stats_data, save_dir):
    labels = stats_data['labels']
    clean_labels = []
    for l in labels:
        if "Average" in l or "AVERAGE" in l:
            clean_labels.append("Average")
        else:
            txt = l.replace('_ens', '').replace('_swa', '').replace('_best', '')
            if "excerpt" in txt:
                txt = txt.replace('excerpt', 'Excerpt ')
            clean_labels.append(txt)

    means = {
        'F1-Score': stats_data['f1_mean'],
        'Precision': stats_data['prec_mean'],
        'Recall': stats_data['rec_mean'],
        'mIoU': stats_data['miou_mean']
    }

    stds = {
        'F1-Score': stats_data['f1_std'],
        'Precision': stats_data['prec_std'],
        'Recall': stats_data['rec_std'],
        'mIoU': stats_data['miou_std']
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
            if "Average" in label or "AVERAGE" in label:
                ax.bar(pos[i], m_vals[i], width, label=name if i == 0 else "", color=color, alpha=0.9,
                       yerr=s_vals[i], capsize=5, ecolor='black')
            else:
                ax.bar(pos[i], m_vals[i], width, label=name if i == 0 else "", color=color, alpha=0.9)

    ax.set_title('Model performance metrics per subject (average of 3 runs)')
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels, fontsize=11, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)

    ax.set_yticks(np.arange(0, 1.1, 0.1))

    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=12,
              framealpha=0.9)

    ax.grid(axis='y', linestyle='--', alpha=0.4)

    if 'Average' in clean_labels:
        idx = labels.index('Average')
        ax.axvspan(idx - 0.5, idx + 0.5, color='gray', alpha=0.1, zorder=0)

    out_file = save_dir / "performance_metrics.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved Metrics Plot to {out_file}")


def main():
    if not STATS_FILE_PATH.exists():
        log.error("Stats file missing.")
        return

    with open(STATS_FILE_PATH, 'r') as f:
        gt_stats = json.load(f)

    load_eval_stats_from_folder(MODEL_RUN_DIR)

    single_repeat_stats = load_single_repeat_stats(MODEL_RUN_DIR, TARGET_REPEAT)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    data_gt_only = prepare_data(gt_stats, {})
    plot_annotation_stats(data_gt_only, PLOTS_DIR)

    if single_repeat_stats:
        data_performance = prepare_data(gt_stats, single_repeat_stats)
        plot_model_performance(data_performance, PLOTS_DIR,
                               folder_name=LOSO_FOLDER_NAME,
                               repeat_name=TARGET_REPEAT)
    else:
        log.warning(f"Skipping performance plot because {TARGET_REPEAT} stats were empty.")

if __name__ == "__main__":
    main()